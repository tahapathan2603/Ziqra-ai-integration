"""
Dataset construction for student training (Part 10).

Independent from model loading: everything here takes an already-built
tokenizer and never imports model_loader.py.

Reads training.data.student_dataset's output directly -- one coach's
already-split, already-validated {"input": {"level1", "level2"},
"target": {...}} records -- as the sole source:

    training/data/datasets/student_dataset/{coach}/{train,validation}.jsonl
            |
            v
    training.prompts.build_conversation_from_student_record(record)
            |  -- fills the coach template from the record's own
            |     already-split level1/level2 and formats its target into
            |     the assistant turn; produces byte-identical `messages`
            |     content to building fresh from a DistillationSample
            |     (verified: see training.data.student_dataset's own
            |     verification, not re-asserted here)
            v
    tokenize (chat template) -> truncate -> mask prompt tokens out of the
    loss -> ConversationDataset

This module used to re-derive conversations by cross-referencing a
combined split file (for session_id/coach membership only) against
training.data.load_coach_samples (for full content) -- two files, two
lookups, for what training.data.student_dataset now already hands over as
one file with everything in it. Reading it directly removes that
cross-referencing entirely and drops this module's dependency on
training.data down to nothing -- it depends only on files on disk plus
training.prompts' formatter.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ..prompts import build_conversation_from_student_record

logger = logging.getLogger(__name__)


class DatasetBuildError(Exception):
    """A student_dataset file is missing or malformed, a record's coach
    doesn't match the config it was loaded under, or a coach's split has
    zero usable examples after tokenization."""


def _read_student_records(path: Path, expected_coach: str) -> List[Dict[str, Any]]:
    """Read one training.data.student_dataset split file. Asserts every
    record's "coach" matches `expected_coach` -- student_dataset writes
    one file per coach, so a mismatch means the wrong file is wired to
    this config, not a data problem to silently tolerate."""
    if not path.exists():
        raise DatasetBuildError(f"Student dataset file not found: {path}. Run training/data/student_dataset.py first.")
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetBuildError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if record.get("coach") != expected_coach:
                raise DatasetBuildError(
                    f"{path}:{lineno}: record's coach={record.get('coach')!r} does not match "
                    f"expected coach {expected_coach!r} -- wrong file wired to this config?"
                )
            records.append(record)
    return records


def _tokenize_conversation(
    tokenizer: PreTrainedTokenizerBase, messages: List[Dict[str, str]], max_length: int
) -> Dict[str, List[int]]:
    """Tokenize one two-turn conversation via the tokenizer's own chat
    template, with the user (prompt) turn's tokens masked out of the loss
    (`labels = -100`) -- the model is only trained to predict the
    assistant's response, never to reproduce the prompt.

    The prompt/assistant boundary is found by rendering the user turn
    alone with `add_generation_prompt=True` (which appends the model's
    turn-opening tokens, e.g. Gemma's "<start_of_turn>model\\n") and
    tokenizing that separately -- its length is exactly where the
    assistant's tokens begin in the full rendering.
    """
    user_messages = messages[:1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100

    # token_type_ids: all-zero, since every example here is plain text (no
    # image tokens -- Gemma3's own convention is 1=image, 0=text).
    #
    # No longer strictly required, but deliberately kept. It was added
    # because Gemma3's MULTIMODAL forward raises
    # "`token_type_ids` is required as a model input when training" without
    # it (confirmed on a real Kaggle run). model_loader.py now narrows
    # multimodal checkpoints to their text tower, and Gemma3TextModel has no
    # such requirement -- verified directly: forward+backward with and
    # without this key produce an identical loss, since the text-only
    # class absorbs it via **kwargs and ignores it, exactly as Qwen2 does.
    # Emitting it unconditionally costs ~8 bytes/token and keeps this
    # collator correct for either loading path, rather than coupling the
    # dataset to which model class the loader happened to pick.
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "token_type_ids": [0] * len(full_ids),
    }


@dataclass(frozen=True)
class ConversationExample:
    """One tokenized, loss-masked training example."""

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    token_type_ids: List[int]


class ConversationDataset(Dataset):
    """A `torch.utils.data.Dataset` over one coach's tokenized
    conversations. Tokenization happens once, eagerly, in `__init__` --
    these datasets are small enough (low thousands of rows) that eager
    tokenization is simpler than an on-the-fly `__getitem__`, with no
    meaningful memory cost.
    """

    def __init__(
        self,
        conversations: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int,
    ) -> None:
        self._examples = self._tokenize_all(conversations, tokenizer, max_sequence_length)

    @staticmethod
    def _tokenize_all(
        conversations: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_sequence_length: int
    ) -> List[ConversationExample]:
        examples = []
        skipped = 0
        for conversation in conversations:
            tokenized = _tokenize_conversation(tokenizer, conversation["messages"], max_sequence_length)
            if all(label == -100 for label in tokenized["labels"]):
                # The assistant's response was truncated away entirely --
                # an all-masked example contributes no loss and no signal.
                skipped += 1
                continue
            examples.append(ConversationExample(**tokenized))
        if skipped:
            logger.warning(
                "Skipped %d/%d example(s) whose assistant response was fully truncated away at "
                "max_sequence_length=%d -- consider raising it.",
                skipped, len(conversations), max_sequence_length,
            )
        return examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        example = self._examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
            "token_type_ids": example.token_type_ids,
        }


@dataclass
class ConversationDataCollator:
    """Pads a batch to its own longest example (dynamic padding) --
    `input_ids`/`attention_mask` pad with the tokenizer's pad token id,
    `labels` pad with -100 so padding never contributes to the loss."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(example["input_ids"]) for example in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels, token_type_ids = [], [], [], []
        for example in batch:
            pad_len = max_len - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(example["attention_mask"] + [0] * pad_len)
            labels.append(example["labels"] + [-100] * pad_len)
            # Pad value doesn't matter semantically (padding is already
            # masked out of the loss via labels=-100, and out of attention
            # via attention_mask=0) -- 0 keeps it consistent with the "text"
            # token type used everywhere else in this all-text dataset.
            token_type_ids.append(example["token_type_ids"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


def build_datasets(
    dataset_cfg,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    project_root: Path,
) -> Tuple[ConversationDataset, ConversationDataset]:
    """Build (train_dataset, validation_dataset) for `dataset_cfg.coach`
    directly from training.data.student_dataset's output.

    Raises:
        DatasetBuildError: a student_dataset file is missing/malformed, a
            record's coach doesn't match `dataset_cfg.coach`, or a split
            ends up with zero usable examples.
    """
    train_path = project_root / dataset_cfg.train_file
    val_path = project_root / dataset_cfg.validation_file

    train_records = _read_student_records(train_path, dataset_cfg.coach)
    val_records = _read_student_records(val_path, dataset_cfg.coach)

    train_conversations = [build_conversation_from_student_record(r) for r in train_records]
    val_conversations = [build_conversation_from_student_record(r) for r in val_records]

    if not train_conversations:
        raise DatasetBuildError(f"No training examples found for coach '{dataset_cfg.coach}' in {train_path}")
    if not val_conversations:
        raise DatasetBuildError(f"No validation examples found for coach '{dataset_cfg.coach}' in {val_path}")

    train_dataset = ConversationDataset(train_conversations, tokenizer, max_sequence_length)
    val_dataset = ConversationDataset(val_conversations, tokenizer, max_sequence_length)
    return train_dataset, val_dataset


__all__ = [
    "ConversationDataCollator",
    "ConversationDataset",
    "ConversationExample",
    "DatasetBuildError",
    "build_datasets",
]
