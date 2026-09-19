"""Verify sub-agent outputs against deterministic rules before synthesis."""


def verify(*args, **kwargs) -> dict:
    """Verify sub-agent results; returns {approved: bool, reasons: list[str]}."""
    raise NotImplementedError
