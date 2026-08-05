"""
Loads accepted evidence packets and derives their Articulation/Delivery
coach packets via the real, unmodified build_coach_packets().

    Level 1 Timeline
    +
    Level 2 Analytics
            |
            v
    build_coach_packets()
            |
      ,-----+-----.
      v           v
Articulation   Delivery
   Packet       Packet

This is the SOLE place in the knowledge-distillation pipeline that performs
this derivation. backend.knowledge_distillation.synthetic_generation no
longer builds or writes Articulation/Delivery packets itself — that inline
logic was removed from its pipeline.py specifically because it duplicated
this module. Synthetic generation's only job is producing raw Level 1 +
Level 2 evidence (packets.jsonl); everything downstream of "turn evidence
into coach packets" lives here.

Nothing here calls a teacher model. This module's own output — the coach
packets — is the *input* the teacher models (MiniMax M3 for Articulation,
MiMo-v2.5 for Delivery) will read in a later stage to produce the actual
coaching output (see docs/coach_output_schema.md). That stage is not part
of this module.
"""

import json
import logging
import os
from typing import Any, Dict, Iterator

from ...feedback.coach_packets import build_coach_packets
from .exceptions import PacketBuildError
from .schemas import CoachPacketPair

logger = logging.getLogger(__name__)


def iter_evidence_packets(packets_path: str) -> Iterator[Dict[str, Any]]:
    """Yield each raw evidence packet dict from a packets.jsonl file, in
    the shape backend.knowledge_distillation.synthetic_generation.schemas.
    SyntheticEvidencePacket.to_dict() produces (session_id/blueprint/
    level1/level2/metadata)."""
    with open(packets_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_packet_pair(evidence_packet: Dict[str, Any]) -> CoachPacketPair:
    """Derive one session's Articulation/Delivery packets from its raw
    evidence, via the real build_coach_packets().

    Raises:
        PacketBuildError: `evidence_packet` is missing session_id/level2,
            or build_coach_packets() itself raised.
    """
    session_id = evidence_packet.get("session_id")
    level2 = evidence_packet.get("level2")
    if not session_id or level2 is None:
        raise PacketBuildError(
            f"Evidence packet is missing session_id and/or level2. Got keys: {list(evidence_packet.keys())}"
        )

    try:
        result = build_coach_packets(level2, session_id=session_id)
    except Exception as e:  # noqa: BLE001 -- any failure here IS the finding
        raise PacketBuildError(f"build_coach_packets() raised for session '{session_id}': {e}") from e

    return CoachPacketPair(session_id=session_id, articulation=result["articulation"], delivery=result["delivery"])


def build_all(packets_path: str) -> Iterator[CoachPacketPair]:
    """Stream a CoachPacketPair for every evidence packet in `packets_path`.

    Raises:
        PacketBuildError: propagated from build_packet_pair for whichever
            packet triggered it; the packet's session_id is in the message.
    """
    for evidence_packet in iter_evidence_packets(packets_path):
        yield build_packet_pair(evidence_packet)


def write_packet_pairs(packets_path: str, out_dir: str) -> int:
    """Build every packet pair from `packets_path` and stream them to
    `out_dir/articulation/articulation.jsonl` and
    `out_dir/delivery/delivery.jsonl` (one line each per session, prefixed
    with session_id — the same shape synthetic_generation used to write
    before this module took over that responsibility).

    Returns:
        Number of packet pairs written.
    """
    articulation_dir = os.path.join(out_dir, "articulation")
    delivery_dir = os.path.join(out_dir, "delivery")
    os.makedirs(articulation_dir, exist_ok=True)
    os.makedirs(delivery_dir, exist_ok=True)

    count = 0
    with open(os.path.join(articulation_dir, "articulation.jsonl"), "w", encoding="utf-8") as art_f, \
            open(os.path.join(delivery_dir, "delivery.jsonl"), "w", encoding="utf-8") as del_f:
        for pair in build_all(packets_path):
            art_f.write(json.dumps({"session_id": pair.session_id, **pair.articulation}, ensure_ascii=False) + "\n")
            del_f.write(json.dumps({"session_id": pair.session_id, **pair.delivery}, ensure_ascii=False) + "\n")
            count += 1
            if count % 500 == 0:
                logger.info("built %d packet pairs", count)

    return count
