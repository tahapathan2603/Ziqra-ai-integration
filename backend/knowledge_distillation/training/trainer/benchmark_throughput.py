"""
Throughput diagnostic for one coach's training config -- NOT part of the
training path itself. Runs a small number of REAL micro-steps (forward +
backward, and at least one real optimizer step) through the actual
`Trainer`/model/dataset this project trains with, instrumented to answer
exactly the questions a "why is this so slow" investigation needs:

  - micro-step (forward+backward) time, separately from
  - optimizer-step time, so that
  - examples/sec and the predicted per-progress-bar-step time can be
    computed from real numbers instead of guessed from the observed
    minutes/step alone.
  - GPU utilization% and memory% sampled throughout (nvidia-smi), to tell
    compute-bound / memory-bound / idle apart.
  - A torch.profiler kernel breakdown, to see whether GPU time is actually
    going into bitsandbytes 4-bit dequant/matmul kernels, attention/MLP
    kernels, or something else entirely.
  - Live confirmation that `use_cache` is really off during training
    (transformers forces this automatically once gradient_checkpointing is
    enabled -- see GradientCheckpointingLayer.__call__ -- this just checks
    it happened rather than trusting that source reading blindly).

Reuses this package's real code throughout -- config loading, model/dataset
construction, and train.py's own `build_training_arguments` -- so the
config actually being benchmarked is the config that would actually run,
not a hand-rolled approximation of it. Writes nothing to the real
checkpoint/output directories: output_dir is redirected to a scratch
subdirectory, and save/eval are both disabled for the duration of the
benchmark.

Usage (same invocation shape as train.py, run from the project root):
    python -m backend.knowledge_distillation.training.trainer.benchmark_throughput \\
        --config backend/knowledge_distillation/training/configs/articulation.yaml \\
        --micro-steps 20
"""

import argparse
import logging
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import torch
from transformers import Trainer

from . import PROJECT_ROOT, ConfigError, load_config
from .dataset import ConversationDataCollator, build_datasets
from .model_loader import load_model_and_tokenizer, select_device
from .train import build_training_arguments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU utilization/memory sampler -- background thread, nvidia-smi polling.
# No-ops cleanly (empty samples, clearly reported as such) if nvidia-smi
# isn't on PATH, e.g. when this is accidentally run on CPU/MPS.
# ---------------------------------------------------------------------------
class GpuSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[dict] = []  # {"util_gpu": %, "util_mem": %, "mem_used_mib": int, "mem_total_mib": int}
        self.available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _poll_loop(self) -> None:
        query = "utilization.gpu,utilization.memory,memory.used,memory.total"
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=True,
                ).stdout.strip()
                # Multi-GPU hosts report one CSV line per device -- this
                # benchmark is meant to run under CUDA_VISIBLE_DEVICES=0
                # (same requirement as the real training run), so only the
                # first line is expected; parse all lines anyway rather
                # than assume, and record the first as "the" GPU.
                line = out.splitlines()[0]
                util_gpu, util_mem, mem_used, mem_total = (int(x.strip()) for x in line.split(","))
                self.samples.append({
                    "util_gpu": util_gpu, "util_mem": util_mem,
                    "mem_used_mib": mem_used, "mem_total_mib": mem_total,
                })
            except Exception as e:  # never let sampling crash the benchmark
                logger.debug("nvidia-smi poll failed: %s", e)
            self._stop.wait(self._interval)

    def start(self) -> None:
        if not self.available:
            logger.warning("nvidia-smi not found -- GPU utilization/memory will NOT be measured (CPU/MPS run?).")
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> str:
        if not self.samples:
            return "  (no samples -- nvidia-smi unavailable or benchmark too short)"
        util_gpu = [s["util_gpu"] for s in self.samples]
        util_mem = [s["util_mem"] for s in self.samples]
        mem_used = [s["mem_used_mib"] for s in self.samples]
        mem_total = self.samples[-1]["mem_total_mib"]
        idle_fraction = sum(1 for u in util_gpu if u == 0) / len(util_gpu)
        return (
            f"  samples: {len(self.samples)}\n"
            f"  GPU compute util%:  min={min(util_gpu)}  mean={statistics.mean(util_gpu):.1f}  max={max(util_gpu)}\n"
            f"  GPU mem-bw util%:   min={min(util_mem)}  mean={statistics.mean(util_mem):.1f}  max={max(util_mem)}\n"
            f"  GPU memory used:    min={min(mem_used)} MiB  mean={statistics.mean(mem_used):.0f} MiB  "
            f"max={max(mem_used)} MiB  (of {mem_total} MiB total)\n"
            f"  fraction of samples at 0% compute util: {idle_fraction:.1%}  "
            f"(high = GPU sitting idle between kernels -- points at CPU-side/dataloader stalls, not compute)"
        )


# ---------------------------------------------------------------------------
# Instrumented Trainer -- times exactly two things transformers doesn't
# expose on its own: one micro-batch's forward+backward (training_step),
# and one real optimizer.step() call. torch.cuda.synchronize() brackets
# both, since CUDA kernel launches are async -- an un-synced Python timer
# would measure launch time, not completion time, and read misleadingly
# fast.
# ---------------------------------------------------------------------------
class _BenchmarkTrainer(Trainer):
    def __init__(self, *args, micro_step_budget: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_step_budget = micro_step_budget
        self.micro_step_times: List[float] = []
        self.optimizer_step_times: List[float] = []
        self._micro_step_count = 0

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Deliberately wrapping AFTER the full optimizer+scheduler pair is
        # built, not inside create_optimizer() alone: torch's own
        # LRScheduler.__init__ (called from create_scheduler, which
        # create_optimizer_and_scheduler calls right after
        # create_optimizer) does its own monkey-patch of optimizer.step to
        # track whether .step() ran before .get_lr() -- and that patch
        # assumes .step is still a normal bound method (it reads
        # .__func__). Wrapping .step with a plain closure before the
        # scheduler gets a chance to do that breaks its patch with
        # `AttributeError: 'function' object has no attribute '__func__'`
        # -- confirmed by actually hitting this locally before fixing it.
        # Wrapping afterward sidesteps the conflict entirely: torch has
        # already done its own patching by the time this replaces .step
        # one more time.
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)
        real_step = self.optimizer.step

        def timed_step(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = real_step(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.optimizer_step_times.append(time.perf_counter() - t0)
            return result

        self.optimizer.step = timed_step

    def training_step(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = super().training_step(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.micro_step_times.append(time.perf_counter() - t0)
        self._micro_step_count += 1
        if self._micro_step_count >= self.micro_step_budget:
            self.control.should_training_stop = True
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="path to a training/configs/*.yaml file")
    parser.add_argument(
        "--micro-steps", type=int, default=20,
        help="how many real micro-batch forward+backward passes to run (default 20 -- covers a full "
             "gradient_accumulation_steps=16 cycle plus a few more, so at least one real optimizer step "
             "gets timed too). The task that asked for this benchmark specifically wanted 5-10 micro-steps "
             "sampled; 20 is kept as the default so an optimizer-step timing is never left empty.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}")
        raise SystemExit(1)
    project_root = args.project_root or PROJECT_ROOT

    device = select_device()
    print(f"Device: {device}")
    if device != "cuda":
        print(
            "WARNING: no CUDA device visible. This benchmark's timings/GPU-utilization numbers are only "
            "meaningful on the real target hardware (Kaggle T4) -- run it there, not locally."
        )

    print(f"Loading model and tokenizer for '{config.dataset.coach}'...")
    model, tokenizer = load_model_and_tokenizer(config.model, config.lora, config.quantization)

    print(f"use_cache before training: {model.config.use_cache!r} "
          f"(gradient_checkpointing_enable + GradientCheckpointingLayer are expected to force this False "
          f"once trainer.train() actually starts stepping -- rechecked below after the first micro-step)")

    print("Building datasets...")
    train_dataset, val_dataset = build_datasets(
        config.dataset, tokenizer, config.training.max_sequence_length, project_root
    )
    print(f"train examples: {len(train_dataset)}  validation examples: {len(val_dataset)}")

    # Real TrainingArguments, from the real code path -- then redirected to
    # a scratch dir and stripped of anything that would write real
    # checkpoints or run real eval during a throughput measurement.
    scratch_output_dir = project_root / "backend/knowledge_distillation/training/outputs" / f"_benchmark_{config.dataset.coach}"
    training_args = build_training_arguments(config, scratch_output_dir)
    training_args.save_strategy = "no"
    training_args.eval_strategy = "no"
    training_args.load_best_model_at_end = False
    training_args.logging_steps = 1  # want a loss line per optimizer step during this short a run
    # Belt-and-suspenders bound on top of the micro-step-count-based stop
    # inside _BenchmarkTrainer -- keeps the loop small even if that signal
    # is ever missed at a boundary.
    training_args.max_steps = (args.micro_steps // config.training.gradient_accumulation_steps) + 2

    data_collator = ConversationDataCollator(tokenizer)
    trainer = _BenchmarkTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator, micro_step_budget=args.micro_steps,
    )

    sampler = GpuSampler(interval_seconds=0.5)
    sampler.start()

    profiler_ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False, with_stack=False,
    )

    print(f"\nRunning {args.micro_steps} real micro-steps (batch_size={config.training.batch_size}, "
          f"gradient_accumulation_steps={config.training.gradient_accumulation_steps})...\n")
    t_wall_start = time.perf_counter()
    with profiler_ctx as prof:
        trainer.train()
    t_wall_total = time.perf_counter() - t_wall_start

    sampler.stop()

    print(f"\nuse_cache after training started: {trainer.model.config.use_cache!r} "
          f"(expect False -- see GradientCheckpointingLayer.__call__ in modeling_layers.py)")

    micro = trainer.micro_step_times
    opt = trainer.optimizer_step_times
    batch_size = config.training.batch_size
    grad_accum = config.training.gradient_accumulation_steps

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    print(f"\nWall time for the whole benchmark ({len(micro)} micro-steps, {len(opt)} optimizer steps): "
          f"{t_wall_total:.2f}s")

    if micro:
        print(f"\nMicro-step (forward+backward, one example, batch_size={batch_size}) time, over {len(micro)} samples:")
        print(f"  mean={statistics.mean(micro):.3f}s  min={min(micro):.3f}s  max={max(micro):.3f}s"
              + (f"  stdev={statistics.stdev(micro):.3f}s" if len(micro) > 1 else ""))
        print(f"  -> examples/sec at the micro-step level: {batch_size / statistics.mean(micro):.3f}")
    else:
        print("\nNo micro-steps captured -- something stopped training before it began (check the log above).")

    if opt:
        print(f"\nOptimizer step (AdamW update over LoRA params only) time, over {len(opt)} samples:")
        print(f"  mean={statistics.mean(opt):.3f}s  min={min(opt):.3f}s  max={max(opt):.3f}s")
    else:
        print(f"\nNo full optimizer step observed -- need at least {grad_accum} micro-steps "
              f"(gradient_accumulation_steps) for one; re-run with --micro-steps >= {grad_accum + 1}.")

    if micro and opt:
        predicted_progress_bar_step = grad_accum * statistics.mean(micro) + statistics.mean(opt)
        effective_batch = batch_size * grad_accum
        print(f"\nPredicted time for ONE progress-bar step ({grad_accum} micro-steps + 1 optimizer step):")
        print(f"  {predicted_progress_bar_step:.2f}s = {predicted_progress_bar_step / 60:.2f} min")
        print(f"  -> effective examples/sec across the full accumulation cycle: "
              f"{effective_batch / predicted_progress_bar_step:.3f}")
        print(f"  Compare this against the observed ~7.5 min/step -- if this predicted number is already "
              f"close to 7.5 min, the bottleneck is genuinely per-micro-step compute/stalling (see below). "
              f"If it's much lower, something OUTSIDE the timed micro-step/optimizer-step window is eating "
              f"the rest (eval, checkpoint save, or the very first step's one-time compile/cache warmup).")

    print("\nGPU utilization / memory (nvidia-smi, sampled every 0.5s throughout):")
    print(sampler.summary())

    if torch.cuda.is_available():
        print("\nTop 15 CUDA kernels by self time (torch.profiler) -- look for bitsandbytes dequant/matmul "
              "kernel names (e.g. containing 'dequantize', 'gemm', 'igemm', 'int4') to gauge bnb 4-bit's "
              "actual share vs plain attention/MLP kernels:")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    else:
        print("\n(no CUDA -- skipping kernel breakdown)")

    print("\nDone. Nothing was written to the real checkpoint/output directories "
          f"(scratch dir used: {scratch_output_dir}, save_strategy=no, eval_strategy=no).")


if __name__ == "__main__":
    main()
