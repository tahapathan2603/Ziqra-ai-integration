"""
Teacher Runner — orchestrates Coach Packet -> Prompt -> Teacher Provider ->
Raw Response for both coaches (Part 4 of teacher_generation).

    Articulation Packet -> build_articulation_prompt() -> generate_articulation() -> MiniMax M3 raw response
    Delivery Packet     -> build_delivery_prompt()     -> generate_delivery()     -> MiMo-v2.5  raw response

This module is pure orchestration: it builds a prompt via prompt_builder and
hands that prompt straight to a TeacherProvider, returning whatever the
provider returns. It does not:

  - build coach packets (packet_builder.py's job)
  - construct prompt text itself (prompt_builder.py's job)
  - call an LLM SDK/client directly (provider.py's job -- no new API
    clients here; see the module-boundary test in
    tests/test_teacher_runner.py)
  - retry on failure (provider.py already does this)
  - parse, validate, or save the response (later, not-yet-built stages)

Retries, timeouts, and non-retryable-error handling all already live in
provider.py (see NonRetryableTeacherError / TeacherProviderError) --
duplicating any of that here would just be two places doing the same job.
"""

from typing import Any, Dict, Optional

from .prompt_builder import build_articulation_prompt, build_delivery_prompt
from .provider import TeacherProvider


class TeacherRunner:
    """Coordinates prompt_builder + provider for both coaches.

    Holds one TeacherProvider so its two underlying clients are reused
    across calls, with the same lazy-config behavior as TeacherProvider
    itself -- constructing a TeacherRunner never touches the network or the
    environment; only a real run_* call does.
    """

    def __init__(self, provider: Optional[TeacherProvider] = None) -> None:
        """
        Args:
            provider: Override for the TeacherProvider, e.g. to inject a
                fake in tests. Defaults to a real TeacherProvider().
        """
        self._provider = provider or TeacherProvider()

    def run_articulation(self, packet: Dict[str, Any], session_id: str) -> str:
        """Build the Articulation prompt from `packet` and send it to
        MiniMax M3.

        Args:
            packet: An articulation coach packet, e.g. a CoachPacketPair's
                `.articulation` (packet_builder.build_packet_pair()'s
                output).
            session_id: The session this packet belongs to.

        Returns:
            The raw response text -- unparsed, unvalidated.

        Raises:
            NonRetryableTeacherError / TeacherProviderError: propagated
                from TeacherProvider.generate_articulation.
        """
        prompt = build_articulation_prompt(packet, session_id)
        return self._provider.generate_articulation(prompt)

    def run_delivery(self, packet: Dict[str, Any], session_id: str) -> str:
        """Build the Delivery prompt from `packet` and send it to
        MiMo-v2.5.

        Args:
            packet: A delivery coach packet, e.g. a CoachPacketPair's
                `.delivery` (packet_builder.build_packet_pair()'s output).
            session_id: The session this packet belongs to.

        Returns:
            The raw response text -- unparsed, unvalidated.

        Raises:
            NonRetryableTeacherError / TeacherProviderError: propagated
                from TeacherProvider.generate_delivery.
        """
        prompt = build_delivery_prompt(packet, session_id)
        return self._provider.generate_delivery(prompt)
