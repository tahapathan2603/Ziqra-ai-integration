"""
Dataset construction for student training (Part 10).

Independent from model loading: everything here takes an already-built
tokenizer and never imports model_loader.py.

Integrates training.data with training.prompts rather than picking one:

    training/data/datasets/splits/{train,validation}.jsonl
            |  (read ONLY for session_id/coach -- which split a session
            |   belongs to; NOT for their baked-in `messages` field)
            v
    training.data.load_coach_samples(coach)     -- re-fetch each full
            |                                       DistillationSample
            v
    training.prompts.build_conversation(sample)  -- format fresh, from
            |                                       the CURRENT templates
            v
    tokenize (chat template) -> truncate -> mask prompt tokens out of the
    loss -> ConversationDataset

Rebuilding conversations fresh (instead of trusting the split files'
baked-in `messages`) means editing articulation_template.txt or
delivery_template.txt takes effect immediately, without re-running
prepare_dataset.py/split_dataset.py first. The split files remain the
single source of truth for train/validation/test *membership* -- that
partitioning is deliberately not redone here.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ..data import load_coach_samples
from ..prompts import build_conversation

logger = logging.getLogger(__name__)


class DatasetBuildError(Exception):
    """A split file is missing, or a coach's split has zero usable
    examples after tokenization."""


def _read_split_session_ids(path: Path, coach: str) -> Set[str]:
    """Which session_ids belong to `coach` in one split file. Only
    `session_id`/`coach` are read -- see module docstring for why the
    file's own `messages` field is deliberately ignored."""
    if not path.exists():
        raise DatasetBuildError(f"Split file not found: {path}. Run training/data/split_dataset.py first.")
    ids: Set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetBuildError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if record.get("coach") == coach and record.get("session_id"):
                ids.add(record["session_id"])
    return ids


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

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


@dataclass(frozen=True)
class ConversationExample:
    """One tokenized, loss-masked training example."""

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


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

        input_ids, attention_mask, labels = [], [], []
        for example in batch:
            pad_len = max_len - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(example["attention_mask"] + [0] * pad_len)
            labels.append(example["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _build_conversations_for_split(coach: str, session_ids: Set[str]) -> List[Dict[str, Any]]:
    """One pass over `load_coach_samples(coach)`, building a fresh
    conversation (via training.prompts) for every sample whose
    session_id is in `session_ids`."""
    conversations = []
    for sample in load_coach_samples(coach):
        if sample.session_id in session_ids:
            conversations.append(build_conversation(sample))
    return conversations


def build_datasets(
    dataset_cfg,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    project_root: Path,
) -> Tuple[ConversationDataset, ConversationDataset]:
    """Build (train_dataset, validation_dataset) for `dataset_cfg.coach`.

    Raises:
        DatasetBuildError: a split file is missing/malformed, or a split
            ends up with zero usable examples.
    """
    train_path = project_root / dataset_cfg.train_file
    val_path = project_root / dataset_cfg.validation_file

    train_ids = _read_split_session_ids(train_path, dataset_cfg.coach)
    val_ids = _read_split_session_ids(val_path, dataset_cfg.coach)

    # Single pass over the coach's samples serves both splits at once,
    # rather than re-joining packet+raw_response data twice.
    train_conversations: List[Dict[str, Any]] = []
    val_conversations: List[Dict[str, Any]] = []
    for sample in load_coach_samples(dataset_cfg.coach):
        if sample.session_id in train_ids:
            train_conversations.append(build_conversation(sample))
        elif sample.session_id in val_ids:
            val_conversations.append(build_conversation(sample))

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
