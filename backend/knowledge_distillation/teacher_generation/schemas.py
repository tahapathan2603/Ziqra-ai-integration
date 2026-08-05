"""
Typed models for teacher generation. `packet_builder.py`'s only output.

`articulation`/`delivery` stay plain dicts deliberately: their shape is
owned by backend.feedback.coach_packets.build_coach_packets(), not this
module — typing them here would duplicate a contract this module doesn't
define and would drift the moment that function's output shape changes.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CoachPacketPair:
    """One session's Articulation + Delivery coach packets, derived from
    its raw evidence via the real build_coach_packets().

    Attributes:
        session_id: The evidence packet's session identifier.
        articulation: build_coach_packets()'s "articulation" value verbatim.
        delivery: build_coach_packets()'s "delivery" value verbatim.
    """

    session_id: str
    articulation: Dict[str, Any]
    delivery: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
