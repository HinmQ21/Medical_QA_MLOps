"""Extract the chosen option letter from raw model output."""

import re

_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
_PHRASE_RE = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:)\s*\(?([A-Za-z])\)?",
    re.IGNORECASE,
)


def parse_answer(text: str, valid_letters: set[str] | None = None) -> str | None:
    """Return the answer letter (uppercase) or None."""
    letter = None
    match = _TAG_RE.search(text)
    if match is None:
        match = _PHRASE_RE.search(text)
    if match is not None:
        letter = match.group(1).upper()
    if letter is not None and valid_letters is not None and letter not in valid_letters:
        return None
    return letter
