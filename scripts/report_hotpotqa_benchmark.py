"""Create a paper-metric report from a complete HotpotQA benchmark run."""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from hotpotqa_benchmark import atomic_json
from research_experiments import paired_bootstrap


PAPER_METRICS = ("em", "f1", "support_recall@2", "support_recall@5")
DISPLAY = {"em": "EM", "f1": "F1", "support_recall@2": "Recall@2", "support_recall@5": "Recall@5"}


def read_manifest(source: Path):
    if source.is_dir():
        matches = list(source.rglob("qa_manifest.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one qa_manifest.json, found {len(matches)}")
        return json.loads(matches[0].read_text(encoding="utf-8"))
    with ZipFile(source) as archive:
        matches = [name for name in archive.namelist() if PurePosixPath(name).name == "qa_manifest.json"]
        if len(matches) != 1:
            raise ValueError(f"Expected one qa_manifest.json, found {len(matches)}")
        return json.load(io.TextIOWrapper(archive.open(matches[0]), encoding="utf-8"))


def load_records(benchmark: Path, methods, expected_ids, allow_incomplete=False):
    result, incomplete = {}, {}
    for method in methods:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in (benchmark / "results" / method).glob("*.json")]
        by_id = {str(record["id"]): record for record in records}
        if len(by_id) != len(records):
            raise ValueError(f"Duplicate result IDs for {method}")
        unexpected = set(by_id) - expected_ids
        if unexpected:
            raise ValueError(f"Unexpected question IDs for {method}: {sorted(unexpected)[:5]}")
        if set(by_id) != expected_ids:
            incomplete[method] = {"completed": len(by_id), "expected": len(expected_ids)}
            if not allow_incomplete or not by_id:
                continue
        result[method] = by_id
    return result, incomplete


def mean_metrics(records):
    available = set.intersection(*(set(record["metrics"]) for record in records.values()))
    return {metric: float(np.mean([record["metrics"][metric] for record in records.values()]))
            for metric in PAPER_METRICS if metric in available}


def error_taxonomy(records):
    counts = Counter()
    examples = defaultdict(list)
    for question_id, record in records.items():
        metrics = record["metrics"]
        if "em" not in metrics:
            kind = "retrieval_only"
        elif metrics["em"] == 1:
            kind = "correct"
        elif metrics.get("support_recall@5", 0) < 1:
            kind = "retrieval_failure"
        else:
            kind = "reader_failure"
        counts[kind] += 1
        if len(examples[kind]) < 10:
            examples[kind].append(question_id)
    return {"counts": dict(counts), "example_question_ids": dict(examples)}


def make_report(benchmark: Path, source: Path, baseline="dense", repetitions=2000, allow_incomplete=False):
    summary_path = benchmark / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = read_manifest(source)
    manifest_by_id = {str(sample["id"]): sample for sample in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("Duplicate IDs in QA manifest")
    methods = list(summary["methods"])
    raw, incomplete = load_records(benchmark, methods, set(manifest_by_id), allow_incomplete)
    if baseline not in raw:
        raise ValueError(f"Complete baseline {baseline!r} is required")

    results = {}
    for method, records in raw.items():
        groups = {}
        for field in ("type", "level"):
            grouped = defaultdict(dict)
            for question_id, record in records.items():
                grouped[str(manifest_by_id[question_id].get(field, "unknown"))][question_id] = record
            groups[f"by_{field}"] = {
                name: {"n": len(items), "metrics": mean_metrics(items)} for name, items in sorted(grouped.items())
            }
        elapsed = [float(record.get("elapsed_seconds", 0)) for record in records.values()]
        results[method] = {
            "n": len(records),
            "metrics": mean_metrics(records),
            **groups,
            "error_taxonomy": error_taxonomy(records),
            "elapsed_seconds": {"mean": float(np.mean(elapsed)), "p95": float(np.quantile(elapsed, 0.95))},
            "diagnostics": summary["methods"].get(method, {}).get("diagnostics", {}),
        }

    comparisons = {}
    for method, records in raw.items():
        if method == baseline or set(records) != set(raw[baseline]):
            continue
        comparisons[method] = {}
        for metric in PAPER_METRICS:
            if metric in results[baseline]["metrics"] and metric in results[method]["metrics"]:
                comparisons[method][metric] = paired_bootstrap(
                    {qid: record["metrics"][metric] for qid, record in raw[baseline].items()},
                    {qid: record["metrics"][metric] for qid, record in records.items()},
                    seed=42,
                    repetitions=repetitions,
                )

    report = {
        "protocol": summary.get("protocol"),
        "questions": len(manifest),
        "baseline": baseline,
        "paper_metrics": list(PAPER_METRICS),
        "warning": "A 100-question pilot is exploratory and is not directly comparable with the paper's 1,000-question run.",
        "results": results,
        "paired_bootstrap_vs_baseline": comparisons,
        "incomplete_excluded": incomplete,
    }
    destination = benchmark / "evaluation_100q"
    atomic_json(destination.with_suffix(".json"), report)

    columns = list(PAPER_METRICS)
    lines = [
        "# HotpotQA English evaluation",
        "",
        report["warning"],
        "",
        "Values are percentages. Recall is document-level Partial Recall, matching the paper protocol.",
        "",
        "| Method | N | " + " | ".join(DISPLAY[key] for key in columns) + " | Retrieval failures | Reader failures | Mean sec/question |",
        "|---|---:|" + "---:|" * (len(columns) + 3),
    ]
    for method, entry in results.items():
        values = [f"{100 * entry['metrics'][key]:.2f}" if key in entry["metrics"] else "—" for key in columns]
        errors = entry["error_taxonomy"]["counts"]
        lines.append(
            f"| {method} | {entry['n']} | " + " | ".join(values)
            + f" | {errors.get('retrieval_failure', 0)} | {errors.get('reader_failure', 0)}"
            + f" | {entry['elapsed_seconds']['mean']:.2f} |"
        )
    lines += [
        "",
        "The JSON report also contains bridge/comparison and difficulty breakdowns, diagnostics, error examples, and paired bootstrap intervals.",
        "",
        f"Incomplete methods excluded: `{json.dumps(incomplete, ensure_ascii=False)}`",
    ]
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination.with_suffix(".md"))
    print(destination.with_suffix(".json"))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="Construction ZIP or graph directory")
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    make_report(args.benchmark.resolve(), args.source.resolve(), args.baseline, args.bootstrap, args.allow_incomplete)
