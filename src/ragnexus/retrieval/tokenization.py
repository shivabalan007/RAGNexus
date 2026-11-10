from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)?", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Small deterministic tokenizer shared by BM25 documents and queries."""
    if not text:
        return []
    return [token.casefold() for token in _TOKEN_RE.findall(text)]
