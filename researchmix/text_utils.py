# researchmix/text_utils.py
import re


def _strip(s: str) -> str:
    return (s or "").strip()


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", _strip(s))


def first_sentence(text: str, max_len: int = 180) -> str:
    """
    Returns the first sentence-ish chunk for card previews.
    """
    t = _clean_ws(text or "")
    if not t:
        return ""
    m = re.search(r"^(.{20,}?[.!?])\s", t)
    if m:
        s = m.group(1).strip()
    else:
        s = t[:max_len].strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "…"
    return s