"""Exception hierarchy for teacher_generation. Every failure raises one of
these, never a bare Exception."""


class TeacherGenerationError(Exception):
    """Base class for every error raised by this module."""


class PacketBuildError(TeacherGenerationError):
    """An evidence packet could not be turned into a CoachPacketPair --
    malformed input (missing session_id/level2) or build_coach_packets()
    itself raised."""


class TeacherProviderError(TeacherGenerationError):
    """A teacher model never produced a usable response — API failure,
    timeout, or an empty body after every retry was exhausted."""


class NonRetryableTeacherError(TeacherProviderError):
    """The provider failed in a way no amount of retrying fixes — an
    exhausted account balance, revoked credentials, or similar account-level
    block. Raised immediately, no backoff loop. A caller orchestrating a
    batch of calls (the future teacher runner) should treat this as a
    reason to stop the whole run, not retry or skip one call — every other
    call to the same model would hit the identical wall. (Learned from a
    real incident in a sibling module: retrying an exhausted-credits error
    ran for 7 hours before being caught.)"""
