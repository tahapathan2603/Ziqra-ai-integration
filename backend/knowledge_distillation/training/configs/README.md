# Training configs

`articulation.yaml` and `delivery.yaml` hold every configurable parameter
for fine-tuning the two student models:

| Config | Student model | Rubrics |
|---|---|---|
| `articulation.yaml` | Gemma (~3B-class) | Pronunciation, MTI |
| `delivery.yaml` | Qwen (~3B-class) | Fluency, Intonation, Engagement |

`training/trainer/` (see its own README/docstrings) is the sole reader of
these files, via `trainer.load_config()`. The values below are sensible
starting points for a ~3B LoRA fine-tune, not tuned results.

Both files share the same structure and field meanings; only the values
differ. Everything below applies to either file.

## `model`

| Field | Meaning |
|---|---|
| `model_name` | Hugging Face Hub id (or local path) of the base model to fine-tune. |
| `tokenizer_name` | Hub id/path for the tokenizer, if it differs from `model_name`. `null` means "use `model_name`'s own tokenizer." |
| `trust_remote_code` | Passed to `from_pretrained(...)`. `false` for both configs — Gemma and Qwen2.5 are natively supported by recent `transformers`, so no custom model code needs to be trusted/executed. |
| `torch_dtype` | Weight/compute precision to load in. `bfloat16` roughly halves memory vs. `float32` and matches both models' native training precision — *if* the GPU actually has bf16 tensor cores. `trainer/model_loader.py`'s `resolve_effective_dtype()` checks this at runtime (`torch.cuda.is_bf16_supported()`) and silently falls back to `float16` when it doesn't — concretely, NVIDIA T4 (Turing, Kaggle's and Colab's free-tier GPU) has no bf16 tensor cores; only Ampere and newer (A100, RTX 30xx+) do. Requesting bf16 on a T4 without this check wouldn't error, it would just run unaccelerated. This value is a request, not a guarantee — check the training run's logs for a fallback warning on unfamiliar hardware. |
| `ollama_model_name` | The same model's Ollama tag (e.g. `"gemma3:4b"`), used **only** as a local-presence check/optional-pull convenience — not the source of the weights actually loaded for training. Ollama stores models as quantized GGUF blobs for its own llama.cpp runtime, which `transformers`/PEFT cannot load for gradient-based fine-tuning; the trainable weights always come from `model_name` via the Hugging Face Hub. See `trainer/model_loader.py`'s docstring. `null` skips the check entirely. |

## `quantization`

4-bit (QLoRA-style) loading via `bitsandbytes`. Requires a CUDA GPU — the trainer detects its absence and falls back to unquantized loading with a warning (this project's reference dev machine is Apple Silicon/MPS, where these fields have no effect).

**This is required, not optional, on a 16GB T4.** Gemma-3's vocabulary is 262,208 tokens, and `transformers`' causal-LM loss does an unchunked `logits.float()`, so an fp16 logits tensor *and* its fp32 copy are alive simultaneously — `batch × seq × 262208 × 6 bytes`. At the longest real training example (4176 tokens) that is 6.6 GB of logits by itself. Unquantized fp16 weights are another 8.6 GB, totalling ~16.3 GB against the T4's 14.56 GiB — it OOM'd. 4-bit brings the weights to ~3.4 GB and the peak to ~11.1 GB.

| Field | Meaning |
|---|---|
| `enabled` | Whether to load the base model 4-bit-quantized. `true` in both configs — see above. When it actually engages (CUDA present *and* bitsandbytes installed), `trainer/model_loader.py` additionally runs `peft.prepare_model_for_kbit_training()`, which casts layernorms/embeddings back to fp32 so the frozen 4-bit base is stable to backprop through. That call is gated on the loaded model's own `is_loaded_in_4bit` flag rather than this field, because this field can be `true` while quantization silently declined on an unsupported host. |
| `load_in_4bit` | Passed to `BitsAndBytesConfig`. |
| `bnb_4bit_compute_dtype` | Precision used for the actual matmuls even though weights are stored 4-bit. |
| `bnb_4bit_quant_type` | `"nf4"` (NormalFloat4, the standard QLoRA choice) or `"fp4"`. |
| `bnb_4bit_use_double_quant` | Quantizes the quantization constants themselves for a small additional memory saving. |

## `dataset`

| Field | Meaning |
|---|---|
| `coach` | Which coach this config is for. `train_file`/`validation_file`/`test_file` are already single-coach files (`training/data/student_dataset.py` writes one per coach), so this isn't a filter — the trainer asserts every record's own `"coach"` field matches it, catching a wrong-file-wired-to-wrong-config mistake early. |
| `train_file` / `validation_file` / `test_file` | Paths to the split JSONL files produced by `training/data/student_dataset.py`. Each record is `{"session_id", "coach", "input": {"level1", "level2"}, "target": {evaluation_analysis, scores, score_reasoning, coach_output, reasoning_trace}}` — the trainer builds a chat-format conversation from it via `training.prompts.build_conversation_from_student_record`. Given as repo-root-relative paths; resolve them against the project root, not the config file's own location. |

## `lora`

Placeholders for the LoRA/PEFT setup the (not-yet-built) trainer will use.

| Field | Meaning |
|---|---|
| `enabled` | Whether to apply LoRA at all vs. full fine-tuning. `true` by default — full fine-tuning a ~3B model is far more memory-hungry for comparable gains here. |
| `rank` | LoRA's low-rank dimension (`r`). Higher = more trainable parameters/capacity, more memory. 16 is a common starting point for ~3B models. |
| `alpha` | LoRA's scaling factor, conventionally set to `2 * rank`. Together with `rank` it controls the effective magnitude of the LoRA update. |
| `dropout` | Dropout applied inside the LoRA adapter layers, for regularization. |
| `target_modules` | Which linear layers get LoRA adapters. The list here (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`) is the standard attention + MLP projection set for both Gemma's and Qwen's (Llama-style) transformer blocks. |

## `training`

| Field | Meaning |
|---|---|
| `epochs` | Full passes over `train_file`. |
| `learning_rate` | Peak LR after warmup. `2e-4` is a conventional LoRA rate — noticeably higher than the `~2e-5` typical of full fine-tuning, since only the small adapter matrices are being trained. |
| `batch_size` | Per-device batch size. **Must stay 1 on a 16GB T4** — the logits term described under `quantization` scales linearly with it, so `batch_size: 2` adds another ~6.6 GB and OOMs even with 4-bit weights. The original `4` was the direct cause of the observed crash (it needed ~25 GB of logits alone). Under multi-GPU (DDP — see below), this is the batch size *each* GPU processes; it does not change based on GPU count. |
| `gradient_accumulation_steps` | Batches accumulated before an optimizer step. Effective batch size = `batch_size * gradient_accumulation_steps * num_gpus` — 16 on one GPU with the defaults, 32 under `accelerate launch --num_processes=2` with the same config. Raised 4 → 16 when `batch_size` dropped 4 → 1, so the effective batch (and therefore the training dynamics) is unchanged at 16. Halve it to 8 if you move to two GPUs and want to hold the effective batch at 16 rather than doubling it. |
| `warmup_ratio` | Fraction of total training steps spent linearly ramping the LR up from 0 to `learning_rate`. |
| `weight_decay` | L2 regularization coefficient for the optimizer. |
| `max_sequence_length` | Truncation cap per training example (prompt + response). **It does not drive memory** — `ConversationDataCollator` pads dynamically to each batch's own longest example, so the real cost is the actual token count, not this ceiling. Measured with each model's *own* tokenizer over all 1600 train rows per coach: articulation/Gemma p50 2830, p95 3954, **max 4176**; delivery/Qwen p50 3594, p95 4011, **max 4309**. Nothing is truncated at 6144, which is kept as free headroom. (An earlier version of this row cited "p99 ≈ 5012, max ≈ 5284" measured with a Llama-family tokenizer — neither model actually being trained — overstating the requirement by ~25%.) Re-measure (`AutoTokenizer.apply_chat_template` + `training.prompts.build_conversation_from_student_record`) if the prompt templates or `teacher_output` schema change. |
| `gradient_checkpointing` | Recomputes activations during backward instead of storing them — roughly 20-30% slower per step, in exchange for a large activation-memory cut. `true` by default. Requires no other action — `trainer/model_loader.py` handles the LoRA-specific prerequisite (`enable_input_require_grads()`) automatically. Necessary but **not sufficient** on a 16GB T4: it only shrinks the per-layer activation term, and the binding constraint here is the logits tensor (see `quantization`), which checkpointing does not touch. An earlier version of this row claimed checkpointing was what made `batch_size: 4` fit on a T4 — it was not, and that configuration OOM'd with checkpointing enabled. |
| `dataloader_num_workers` / `dataloader_persistent_workers` / `dataloader_prefetch_factor` | Background data-loading workers, whether they persist across epochs (avoids respawning), and how many batches each stages ahead of GPU compute. The benefit is modest for this dataset specifically — tokenization happens once, eagerly, at dataset-construction time (`trainer/dataset.py`), so there's little per-batch CPU work left to overlap with GPU compute. `persistent_workers`/`prefetch_factor` are automatically ignored (not passed to `TrainingArguments`) if `dataloader_num_workers` is 0. |
| `eval_accumulation_steps` | How many eval batches of raw logits Trainer holds on-device before offloading to CPU. `null` accumulates everything, which is fine for this dataset's small (~200-sample) eval sets; set a small integer if evaluation ever OOMs on a larger one. |
| `early_stopping_patience` | Stop training if validation loss hasn't improved for this many evaluation calls in a row. `null` disables early stopping. |

### Multi-GPU (e.g. Kaggle T4×2)

Nothing in this config changes for multi-GPU — the same YAML drives both a single-GPU and a multi-GPU run. What changes is how `train.py` is *launched*:

```bash
# single GPU / CPU / MPS -- one process.
# CUDA_VISIBLE_DEVICES=0 is REQUIRED on any host with more than one GPU, and
# must be set in the environment BEFORE python starts -- see below.
CUDA_VISIBLE_DEVICES=0 \
python -m backend.knowledge_distillation.training.trainer.train --config training/configs/articulation.yaml

# two GPUs -- data-parallel via Hugging Face Accelerate (DistributedDataParallel
# under the hood, not the older, single-process DataParallel)
accelerate launch --multi_gpu --num_processes=2 \
    -m backend.knowledge_distillation.training.trainer.train --config training/configs/articulation.yaml
```

`transformers.Trainer` detects the multi-process launch automatically and wraps the model in DDP itself — nothing in this codebase manually constructs a `DistributedDataParallel`. Each process gets a full model replica and a different batch slice; gradients sync after each backward pass. This is a **throughput** optimization — it requires the full model to already fit on one GPU, which is what `gradient_checkpointing` (above) and a hardware-appropriate `torch_dtype` (see `model` section — T4 has no bf16 tensor cores, so bf16 requests silently fall back to fp16 on it) are for. Two T4s does **not** mean 32GB of usable model memory; each GPU still needs the model to fit in its own 16GB.

### Running single-process on a multi-GPU host — you must set `CUDA_VISIBLE_DEVICES`

`transformers.Trainer` silently wraps the model in the legacy `torch.nn.DataParallel` the moment more than one CUDA device is visible to a single, non-distributed process — the exact thing this multi-GPU support exists to avoid. It multiplies the effective batch size by the GPU count and gathers every replica's outputs onto GPU 0.

**This cannot be fixed from inside the training process, and an earlier attempt to do so failed silently.** `torch.cuda.device_count()` caches its result in a module global as soon as CUDA is initialised and never re-reads the environment afterwards — and CUDA is already initialised during *imports* (the bitsandbytes/torchao banner prints before `train.py`'s first log line). Setting `CUDA_VISIBLE_DEVICES` inside `main()` is therefore too late. On two live Kaggle T4×2 runs the "pinning" log line printed and DataParallel engaged anyway; the tell was the step count — 1600 rows at effective batch 16 over 3 epochs is **300** steps on one GPU, but both runs reported **150**, i.e. Trainer had doubled `train_batch_size` across two GPUs.

So `train.py` now *asserts* rather than fixes: `_assert_not_data_parallel()` raises `MultiGPUConfigurationError` up front (before the multi-GB model download) if more than one GPU is visible outside a distributed launch. Set the variable in the environment before starting Python — `CUDA_VISIBLE_DEVICES=0 python -m ...`, or `%env CUDA_VISIBLE_DEVICES=0` in a notebook cell, as `notebooks/train_on_kaggle.ipynb` Cell 6 does. Under `accelerate launch` the assertion skips itself (it reads `WORLD_SIZE`), and you must *not* pin the variable there — each process needs its own GPU.

**Step count is the quickest check that it worked:** 300, not 150.

Every print and file write in `train.py` is already guarded to the main process (rank 0), so a multi-GPU run doesn't duplicate terminal output or race on the adapter directory.

## `checkpointing`

| Field | Meaning |
|---|---|
| `output_dir` | Root directory for one training run. The trainer creates three subdirectories underneath it — `checkpoints/` (intermediate, periodic), `adapters/` (the final LoRA adapter only, written once at the end), `logs/` — so the final adapter is never mixed in with intermediate checkpoints. Kept separate per coach so the two runs never overwrite each other. |
| `save_strategy` | `"epoch"` (save at the end of every epoch) or `"steps"` (save every `save_steps`). |
| `save_steps` | Only used when `save_strategy` is `"steps"`. |
| `logging_steps` | How often (in steps) to log training metrics. |
| `evaluation_strategy` | `"epoch"` or `"steps"`, same shape as `save_strategy` but for running evaluation against `validation_file`. (Internally passed to `transformers.TrainingArguments` as `eval_strategy` — that library renamed the argument; this config's field name is unaffected.) |
| `eval_steps` | Only used when `evaluation_strategy` is `"steps"`. |

## `randomness`

| Field | Meaning |
|---|---|
| `seed` | Seeds shuffling/initialization for reproducibility. Set to 42 to match `training/data/split_dataset.py`'s default split seed — not required to match, but keeping them aligned makes the whole pipeline's randomness traceable to one number. |

## Editing these files

- Both YAML files must stay structurally identical (same section names,
  same field names) — only values should differ between coaches. A
  future config-loading utility can then treat them interchangeably.
- Keep `target_modules` as a YAML list (not a comma-separated string) —
  it maps directly to PEFT's `LoraConfig(target_modules=[...])`.
- `tokenizer_name: null` and any other `null` value loads as Python
  `None` via PyYAML — treat that as "use the default," not "unset/invalid."
