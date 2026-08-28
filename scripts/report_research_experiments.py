"""Summarize COMPLETE, matched research arms; never silently compare partial runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hotpotqa_benchmark import atomic_json, digest
from research_experiments import paired_bootstrap


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def report(root, stage="diagnostic", split="smoke", baseline="dense", repetitions=2000):
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    protocol = read_json(root / "protocol.json")
    expected = set(protocol["splits"][split])
    if not expected:
        raise ValueError("Empty evaluation split")
    results, diagnostics, incomplete, raw = {}, {}, {}, {}
    for folder in sorted((root / stage / split).glob("*")):
        if not folder.is_dir() or not (folder / "config.json").exists():
            continue
        config = read_json(folder / "config.json")
        if (config["protocol_id"] != digest(protocol) or set(config["question_ids"]) != expected
                or config["stage"] != stage or config["split"] != split):
            raise ValueError(f"Incomparable run: {folder}")
        records = [read_json(p) for p in (folder / "results").glob("*.json")]
        if any(r["config_id"] != digest(config) for r in records):
            raise ValueError("Record configuration mismatch")
        ids = [str(r["id"]) for r in records]
        if len(set(ids)) != len(ids) or set(ids) - expected:
            raise ValueError("Unexpected or duplicate question IDs")
        if set(ids) != expected:
            incomplete[folder.name] = {"completed": len(ids), "expected": len(expected)}
            continue
        metric_keys = set(records[0]["metrics"])
        if any(set(r["metrics"]) != metric_keys for r in records):
            raise ValueError("Mixed metric sets in one arm")
        means = {k: float(np.mean([r["metrics"][k] for r in records])) for k in sorted(metric_keys)}
        groups = {}
        for kind in sorted({r["question_type"] for r in records}):
            subset = [r for r in records if r["question_type"] == kind]
            groups[kind] = {"n": len(subset), "metrics": {k: float(np.mean([r["metrics"][k] for r in subset])) for k in sorted(metric_keys)}}
        timings = {k: {"mean": float(np.mean([r["timing"][k] for r in records])),
                       "p95": float(np.quantile([r["timing"][k] for r in records], 0.95))}
                   for k in ("retrieval_seconds", "answer_seconds", "total_seconds")}
        usages = [info["usage"] for r in records for info in (r.get("reader"), r["retrieval_trace"].get("filter"))
                  if info and info.get("usage")]
        llm_calls = sum(int(r.get("reader") is not None) + int(r["retrieval_trace"].get("filter", {}).get("enabled", False)) for r in records)
        entry = {"n": len(records), "metrics": means, "by_question_type": groups, "timing_seconds": timings,
                 "llm_usage": {"calls": llm_calls, "calls_with_usage": len(usages),
                    "reported_total_tokens": sum(u["total_tokens"] for u in usages) if usages else None},
                 "dense_fallback_questions": sum(bool(r["retrieval_trace"].get("dense_fallback")) for r in records),
                 "filter_candidates_dropped_for_context": sum(r["retrieval_trace"].get("filter", {}).get("dropped_for_context", 0) for r in records),
                 "reader_truncated_passages": sum((r.get("reader") or {}).get("truncated_passages", 0) for r in records),
                 "sessions": read_json(folder / "sessions.json")}
        if config["arm"]["method"] in {"oracle", "no_context"}:
            diagnostics[folder.name] = entry
        else:
            results[folder.name] = entry
            raw[folder.name] = records
    comparisons = {}
    if baseline in raw:
        for name, records in raw.items():
            if name == baseline:
                continue
            common = set(results[name]["metrics"]) & set(results[baseline]["metrics"])
            comparisons[name] = {k: paired_bootstrap(
                {r["id"]: r["metrics"][k] for r in raw[baseline]},
                {r["id"]: r["metrics"][k] for r in records}, protocol["settings"]["seed"], repetitions)
                for k in sorted(common)}
    result = {"stage": stage, "split": split, "baseline": baseline,
              "warning": "Smoke is debugging only. Dev comparisons are exploratory. CIs assume independent questions; shared documents/multiple comparisons are not corrected.",
              "results": results, "reader_diagnostics_not_competitive": diagnostics,
              "incomplete_excluded": incomplete, "paired_vs_baseline": comparisons,
              "baseline_available": baseline in raw}
    destination = root / f"report_{stage}_{split}"
    atomic_json(destination.with_suffix(".json"), result)
    columns = ["support_recall@2", "all_support@5", "support_recall@10", "em", "f1"]
    lines = [f"# Research report: {stage} / {split}", "", result["warning"], "",
             "Values below are percentages. Only complete runs on identical question IDs are included.", "",
             "| Arm | N | " + " | ".join(columns) + " | Mean query s |",
             "|---|---:|" + "---:|" * (len(columns) + 1)]
    for name, entry in results.items():
        values = [f"{100 * entry['metrics'][k]:.2f}" if k in entry["metrics"] else "—" for k in columns]
        lines.append(f"| {name} | {entry['n']} | " + " | ".join(values) + f" | {entry['timing_seconds']['total_seconds']['mean']:.3f} |")
    lines += ["", "Oracle/no-context results, paired bootstrap intervals, type breakdowns, LLM usage and setup sessions are in the JSON report.",
              "Setup/cache timings are not cold-build comparisons. Post-hoc pruning does not save already-spent extraction/concept-generation cost.", "",
              f"Incomplete arms excluded: {json.dumps(incomplete)}", f"Complete baseline available: {baseline in raw}"]
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination.with_suffix(".md"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=("diagnostic", "qa"), default="diagnostic")
    parser.add_argument("--split", choices=("smoke", "dev", "test"), default="smoke")
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report(args.root, args.stage, args.split, args.baseline, args.bootstrap)
