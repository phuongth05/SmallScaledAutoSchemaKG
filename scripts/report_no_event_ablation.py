#!/usr/bin/env python3
"""Write a paired Full vs No-Event HotpotQA benchmark comparison table.

Inputs are the two work directories produced by ``run_hotpotqa_vn.py``.  The
script reads saved summaries only; it never loads a model or calls an LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    ("Recall@2", "support_recall@2"), ("Recall@5", "support_recall@5"),
    ("Recall@10", "support_recall@10"), ("All-support Hit@2", "all_support@2"),
    ("All-support Hit@5", "all_support@5"), ("All-support Hit@10", "all_support@10"),
    ("QA EM", "em"), ("QA F1", "f1"),
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def construction_stats(work: Path) -> dict:
    summary = read_json(work / "graph" / "run_summary.json")
    graph = work / "graph" / "kg_graphml" / "hotpotqa_corpus_graph.graphml"
    types = summary.get("node_types", {})
    return {
        "Extraction time (s)": (read_json(work / "graph" / "extraction_progress.json")
                                .get("extraction_seconds")),
        "LLM calls": summary.get("extraction", {}).get("llm_calls"),
        "LLM total tokens": summary.get("extraction", {}).get("token_usage", {}).get("total_tokens"),
        "Nodes": summary.get("nodes"), "Edges": summary.get("edges"),
        "Entities": types.get("entity", 0), "Events": types.get("event", 0),
        "Concepts": types.get("concept", 0),
        "GraphML size (bytes)": graph.stat().st_size if graph.is_file() else None,
    }


def benchmark_metrics(work: Path, variant: str) -> dict:
    summary = read_json(work / "benchmark" / "summary.json")
    method = summary.get("methods", {}).get(variant)
    if not method or not method.get("complete"):
        raise ValueError(f"Incomplete or missing benchmark variant {variant!r} in {work}")
    return method.get("metrics", {})


def fmt(value):
    return "—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_work_dir", type=Path)
    parser.add_argument("no_event_work_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Default: <no-event-work-dir>/no_event_comparison.md")
    args = parser.parse_args(argv)
    full, no_event = args.full_work_dir.resolve(), args.no_event_work_dir.resolve()
    full_metrics, no_metrics = benchmark_metrics(full, "full"), benchmark_metrics(no_event, "no_event")
    rows = []
    for label, key in METRICS:
        left, right = full_metrics.get(key), no_metrics.get(key)
        if left is not None or right is not None:
            rows.append((label, left, right))
    for label, left, right in ((key, construction_stats(full)[key], construction_stats(no_event)[key])
                               for key in construction_stats(full)):
        rows.append((label, left, right))
    lines = ["# Full vs No-Event AutoSchemaKG", "", "| Metric | Full | No-Event | Difference (No-Event − Full) |",
             "|---|---:|---:|---:|"]
    for label, left, right in rows:
        delta = right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
        lines.append(f"| {label} | {fmt(left)} | {fmt(right)} | {fmt(delta)} |")
    output = args.output or no_event / "no_event_comparison.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
