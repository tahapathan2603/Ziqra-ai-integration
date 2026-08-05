"""
Prompt templates for student training (Part 9).

Turns a training.data.DistillationSample into a complete Hugging Face
chat-format training conversation, via a coach-specific instruction
template (`{coach}_template.txt` in this directory) plus that sample's own
evidence and teacher output. See prompt_builder.py's docstring for the
full Sample -> conversation pipeline.

Independent of the (not-yet-implemented) trainer: nothing here imports
torch, transformers, or peft -- only training.data and the standard
library.

Usage:
    from backend.knowledge_distillation.training.prompts import build_conversations

    for conversation in build_conversations(coach="articulation"):
        ...  # conversation == {"session_id", "coach", "messages": [...]}
"""

from .prompt_builder import (
    PromptBuildError,
    build_assistant_turn,
    build_conversation,
    build_conversations,
    build_user_turn,
    load_template,
    split_evidence,
)

__all__ = [
    "PromptBuildError",
    "build_assistant_turn",
    "build_conversation",
    "build_conversations",
    "build_user_turn",
    "load_template",
    "split_evidence",
]
