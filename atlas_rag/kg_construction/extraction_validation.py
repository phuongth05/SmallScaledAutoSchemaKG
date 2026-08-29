"""Lightweight semantic guards applied after structural JSON validation."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


VI_EVENT_RELATIONS = {
    "trước": ("trước", "trước khi"),
    "sau": ("sau", "sau khi"),
    "cùng thời điểm": ("cùng thời điểm", "đồng thời", "trong khi"),
    "bởi vì": ("bởi vì", "do đó", "nhờ", "vì"),
    "dẫn đến": ("dẫn đến", "khiến", "làm cho", "do đó", "kết quả", "nên"),
}
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "các", "của", "cho", "đã", "đang", "được", "là", "một", "những",
    "này", "năm", "ở", "sự", "tại", "theo", "trong", "và", "vào", "với",
}


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return _SPACE.sub(" ", " ".join(_WORD.findall(text))).strip()


def _content_tokens(value: Any) -> set[str]:
    return {token for token in _normalize(value).split() if token not in _STOPWORDS}


def _grounded(event: str, source: str) -> bool:
    event_tokens = _content_tokens(event)
    if len(event_tokens) < 2:
        return False
    source_tokens = _content_tokens(source)
    return len(event_tokens & source_tokens) / len(event_tokens) >= 0.6


def sanitize_vietnamese_event_relations(
    items: list[dict[str, Any]], source_text: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Keep only explicit, grounded Vietnamese temporal/causal event edges."""
    source = _normalize(source_text)
    kept: list[dict[str, str]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        head = str(item.get("Head", "")).strip()
        relation = _normalize(item.get("Relation", ""))
        tail = str(item.get("Tail", "")).strip()
        reason = None
        if relation not in VI_EVENT_RELATIONS:
            reason = "relation_not_allowed"
        elif _normalize(head) == _normalize(tail):
            reason = "self_loop"
        elif len(_normalize(head).split()) < 3 or len(_normalize(tail).split()) < 3:
            reason = "endpoint_not_event_clause"
        elif not _grounded(head, source) or not _grounded(tail, source):
            reason = "endpoint_not_grounded"
        elif not any(cue in source for cue in VI_EVENT_RELATIONS[relation]):
            reason = "relation_not_explicit_in_source"
        triple = (_normalize(head), relation, _normalize(tail))
        if reason is None and triple in seen:
            reason = "semantic_duplicate"
        if reason:
            dropped.append({"reason": reason, "item": item})
            continue
        seen.add(triple)
        kept.append({"Head": head, "Relation": relation, "Tail": tail})
    return kept, dropped
