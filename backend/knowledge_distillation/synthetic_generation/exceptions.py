"""
Exception hierarchy for synthetic evidence generation. Every failure raises
one of these, never a bare Exception, so pipeline.py can tell a
recoverable per-blueprint failure apart from a reason to stop the whole run.
"""


class SyntheticGenerationError(Exception):
    """Base class for every error raised by this module."""


class ProviderError(SyntheticGenerationError):
    """The provider never produced a usable response — API failure,
    timeout, or an empty body after every retry was exhausted."""


class NonRetryableProviderError(ProviderError):
    """The provider failed in a way no amount of retrying fixes — an
    exhausted account balance, revoked credentials, or similar account-level
    block. Raised immediately, no backoff loop. pipeline.py stops the whole
    run on this, not just the current blueprint, since every other call
    would hit the identical wall."""


class EvidenceParsingError(SyntheticGenerationError):
    """The provider responded, but no valid JSON with the required
    top-level {level1, level2} shape could be extracted from it."""
