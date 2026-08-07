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

Optional 4-bit (QLoRA-style) loading via `bitsandbytes`. Requires a CUDA GPU — the trainer detects its absence and falls back to unquantized loading with a warning (this project's reference dev machine is Apple Silicon/MPS, where these fields have no effect).

| Field | Meaning |
|---|---|
| `enabled` | Whether to load the base model 4-bit-quantized. `false` by default. |
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
| `batch_size` | Per-device batch size. Lower this first if you hit an out-of-memory error. Under multi-GPU (DDP — see below), this is the batch size *each* GPU processes; it does not change based on GPU count. |
| `gradient_accumulation_steps` | Batches accumulated before an optimizer step. Effective batch size = `batch_size * gradient_accumulation_steps * num_gpus` — 16 on one GPU with the defaults, 32 under `accelerate launch --num_processes=2` with the same config. Lower this to 2 if you want to hold the effective batch at 16 under two GPUs instead of doubling it. |
| `warmup_ratio` | Fraction of total training steps spent linearly ramping the LR up from 0 to `learning_rate`. |
| `weight_decay` | L2 regularization coefficient for the optimizer. |
| `max_sequence_length` | Token cap per training example (prompt + response). 6144 was chosen by tokenizing the *actual* rendered chat template (`trainer/dataset.py`'s own tokenization path, not a char-count estimate) over all 4000 samples with a real Llama-family tokenizer: p99 ≈ 5012 tokens, max ≈ 5284. Re-measure this (`AutoTokenizer.apply_chat_template` + `training.prompts.build_conversation`) if the prompt templates or `teacher_output` schema change — the jump from 4096 came from adding `evaluation_analysis`/`score_reasoning` to every teacher_output, which nearly doubled the assistant turn's length. |
| `gradient_checkpointing` | Recomputes activations during backward instead of storing them — roughly 20-30% slower per step, but a large activation-memory cut (this is what makes `batch_size: 4` at `max_sequence_length: 6144` actually fit on a 16GB T4; the identical config OOM'd without it). `true` by default. Requires no other action — `trainer/model_loader.py` handles the LoRA-specific prerequisite (`enable_input_require_grads()`) automatically. |
| `dataloader_num_workers` / `dataloader_persistent_workers` / `dataloader_prefetch_factor` | Background data-loading workers, whether they persist across epochs (avoids respawning), and how many batches each stages ahead of GPU compute. The benefit is modest for this dataset specifically — tokenization happens once, eagerly, at dataset-construction time (`trainer/dataset.py`), so there's little per-batch CPU work left to overlap with GPU compute. `persistent_workers`/`prefetch_factor` are automatically ignored (not passed to `TrainingArguments`) if `dataloader_num_workers` is 0. |
| `eval_accumulation_steps` | How many eval batches of raw logits Trainer holds on-device before offloading to CPU. `null` accumulates everything, which is fine for this dataset's small (~200-sample) eval sets; set a small integer if evaluation ever OOMs on a larger one. |
| `early_stopping_patience` | Stop training if validation loss hasn't improved for this many evaluation calls in a row. `null` disables early stopping. |

### Multi-GPU (e.g. Kaggle T4×2)

Nothing in this config changes for multi-GPU — the same YAML drives both a single-GPU and a multi-GPU run. What changes is how `train.py` is *launched*:

```bash
# single GPU / CPU / MPS -- one process
python -m backend.knowledge_distillation.training.trainer.train --config training/configs/articulation.yaml

# two GPUs -- data-parallel via Hugging Face Accelerate (DistributedDataParallel
# under the hood, not the older, single-process DataParallel)
accelerate launch --multi_gpu --num_processes=2 \
    -m backend.knowledge_distillation.training.trainer.train --config training/configs/articulation.yaml
```

`transformers.Trainer` detects the multi-process launch automatically and wraps the model in DDP itself — nothing in this codebase manually constructs a `DistributedDataParallel`. Each process gets a full model replica and a different batch slice; gradients sync after each backward pass. This is a **throughput** optimization — it requires the full model to already fit on one GPU, which is what `gradient_checkpointing` (above) and a hardware-appropriate `torch_dtype` (see `model` section — T4 has no bf16 tensor cores, so bf16 requests silently fall back to fp16 on it) are for. Two T4s does **not** mean 32GB of usable model memory; each GPU still needs the model to fit in its own 16GB.

Every print and file write in `train.py`/`callbacks.py` is already guarded to run on the main process only (rank 0), so a multi-GPU run doesn't produce duplicated terminal output or racing writes to the same adapter directory — this is handled automatically, not something you need to configure.

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
