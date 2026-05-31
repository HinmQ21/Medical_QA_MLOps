"""Resolve the compute device for the retrieval encoder."""

import os


def resolve_device(explicit: str | None = None) -> str:
    """Return explicit device, env RETRIEVAL_DEVICE, or CPU."""
    if explicit:
        return explicit
    return os.environ.get("RETRIEVAL_DEVICE", "cpu")
