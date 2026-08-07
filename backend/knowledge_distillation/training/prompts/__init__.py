"""
Prompt templates for student training (Part 9).

Turns evidence + teacher output into a complete Hugging Face chat-format
training conversation, via a coach-specific instruction template
(`{coach}_template.txt` in this directory). Two entry points, both ending
in the same {"session_id", "coach", "messages": [...]} shape:

    build_conversation(sample)               -- from a training.data
                                                 DistillationSample; derives
                                                 the Level 1/Level 2 split
                                                 fresh via split_evidence()
    build_conversation_from_student_record(r) -- from a training.data.
                                                 student_dataset record,
                                                 which already has that
                                                 split precomputed

training.trainer.dataset.py uses the second: it reads
training.data.student_dataset's *.jsonl files directly rather than
re-joining coach packets and raw responses itself.

Independent of the trainer: nothing here imports torch, transformers, or
peft -- only training.data and the standard library.

Usage:
    from backend.knowledge_distillation.training.prompts import build_conversations

    for conversation in build_conversations(coach="articulation"):
        ...  # conversation == {"session_id", "coach", "messages": [...]}
"""

from .prompt_builder import (
    PromptBuildError,
    build_assistant_turn,
    build_conversation,
    build_conversation_from_student_record,
    build_conversations,
    build_user_turn,
    load_template,
    split_evidence,
)

__all__ = [
    "PromptBuildError",
    "build_assistant_turn",
    "build_conversation",
    "build_conversation_from_student_record",
    "build_conversations",
    "build_user_turn",
    "load_template",
    "split_evidence",
]
