"""Utilities for handling large, unstructured source documents."""
import re


def split_document(text: str, max_chars: int = 6000) -> list[str]:
    """Split on paragraph/sentence/word boundaries without dropping source content."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        boundary = max_chars
        candidates = [
            (remaining.rfind("\n\n", 0, max_chars), 2),
            (remaining.rfind("\n", 0, max_chars), 1),
            (remaining.rfind(". ", 0, max_chars), 2),
            (remaining.rfind(" ", 0, max_chars), 1),
        ]
        for candidate, delimiter_length in candidates:
            if candidate >= max_chars // 2:
                boundary = candidate + delimiter_length
                break
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def representative_text(chunks: list[str], max_chars: int = 9000) -> str:
    """Return balanced beginning/middle/ending evidence within the requested budget."""
    if not chunks:
        return ""
    if len(chunks) <= 3:
        selected = chunks
    else:
        selected = [chunks[0], chunks[len(chunks) // 2], chunks[-1]]
    separator = "\n\n[DOCUMENT SECTION]\n\n"
    budget_per_section = max(1, (max_chars - len(separator) * (len(selected) - 1)) // len(selected))
    excerpts = []
    for section in selected:
        if len(section) <= budget_per_section:
            excerpts.append(section)
            continue
        marker = "\n[… excerpt …]\n"
        content_budget = max(1, budget_per_section - len(marker))
        head_size = (content_budget + 1) // 2
        tail_size = content_budget - head_size
        excerpts.append(section[:head_size] + marker + (section[-tail_size:] if tail_size else ""))
    return separator.join(excerpts)[:max_chars]


def clean_output(text: str) -> str:
    """Remove surrounding Markdown fences before downstream JSON consolidation."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned
