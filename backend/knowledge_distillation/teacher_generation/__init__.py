"""
Teacher generation:

  1. packet_builder — derives Articulation/Delivery coach packets from
     accepted synthetic evidence, via the real build_coach_packets().
  2. provider — the Teacher Communication Layer: sends a prompt to MiniMax
     M3 (Articulation) or MiMo-v2.5 (Delivery) and returns the raw response.
  3. prompt_builder — turns a coach packet into the prompt text sent to
     that coach's teacher model.
  4. teacher_runner — orchestrates packet -> prompt -> provider -> raw
     response for both coaches. Pure orchestration: no parsing, no
     validation, no saving.

    synthetic_generation's packets.jsonl (Level 1 + Level 2, raw evidence)
            |
            v
    packet_builder.build_all() / write_packet_pairs()
            |
            v
    CoachPacketPair (Articulation packet + Delivery packet)
            |
            v
    prompt_builder.build_articulation_prompt(packet, session_id)
    prompt_builder.build_delivery_prompt(packet, session_id)
            |
            v
    provider.TeacherProvider().generate_articulation(prompt)   -> MiniMax M3
    provider.TeacherProvider().generate_delivery(prompt)        -> MiMo-v2.5

`provider.py` is communication only — it does not build prompts, does not
know about coach packets/Level 1/Level 2, and does not parse or validate a
response. `prompt_builder.py` is prompt construction only — it does not call
a teacher model and does not build coach packets; it imports neither
`provider.py` nor `packet_builder.py`. The actual teacher execution flow and
response validation are later, not-yet-implemented modules that will call
`prompt_builder.py` and `provider.py` together as their input/transport.

Usage:
    from backend.knowledge_distillation.teacher_generation import write_packet_pairs

    count = write_packet_pairs(
        packets_path="backend/knowledge_distillation/synthetic_generation/datasets/packets/packets.jsonl",
        out_dir="backend/knowledge_distillation/teacher_generation/datasets",
    )

    from backend.knowledge_distillation.teacher_generation import TeacherRunner

    runner = TeacherRunner()
    for pair in build_all(packets_path):
        raw_articulation = runner.run_articulation(pair.articulation, pair.session_id)
        raw_delivery = runner.run_delivery(pair.delivery, pair.session_id)
"""

from .claude_teacher import ClaudeTeacherProvider
from .config import ARTICULATION_ENV_PREFIX, DELIVERY_ENV_PREFIX, TeacherGenerationConfig
from .exceptions import (
    NonRetryableTeacherError,
    PacketBuildError,
    TeacherGenerationError,
    TeacherProviderError,
)
from .packet_builder import build_all, build_packet_pair, iter_evidence_packets, write_packet_pairs
from .prompt_builder import (
    ARTICULATION_RUBRICS,
    ARTICULATION_TEACHER,
    DELIVERY_RUBRICS,
    DELIVERY_TEACHER,
    build_articulation_prompt,
    build_delivery_prompt,
)
from .provider import TeacherModelClient, TeacherProvider
from .schemas import CoachPacketPair
from .teacher_runner import TeacherRunner

__all__ = [
    "ARTICULATION_ENV_PREFIX",
    "ARTICULATION_RUBRICS",
    "ARTICULATION_TEACHER",
    "ClaudeTeacherProvider",
    "CoachPacketPair",
    "DELIVERY_ENV_PREFIX",
    "DELIVERY_RUBRICS",
    "DELIVERY_TEACHER",
    "NonRetryableTeacherError",
    "PacketBuildError",
    "TeacherGenerationConfig",
    "TeacherGenerationError",
    "TeacherModelClient",
    "TeacherProvider",
    "TeacherProviderError",
    "TeacherRunner",
    "build_all",
    "build_articulation_prompt",
    "build_delivery_prompt",
    "build_packet_pair",
    "iter_evidence_packets",
    "write_packet_pairs",
]
