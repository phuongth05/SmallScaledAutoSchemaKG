"""Audit a resumable HotpotQA-VN extraction pilot without calling an LLM."""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path


STAGES = ("entity_relation", "event_entity", "event_relation")
WARNING_PATTERNS = {
    "missing_field_items": re.compile(r"missing required keys:"),
    "duplicate_items": re.compile(r"is a duplicate triple:"),
    "empty_value_items": re.compile(r"\bis empty; dropping item:"),
    "null_value_items": re.compile(r"\bis null; dropping item:"),
    "invalid_array_items": re.compile(r"must be an array; dropping item:"),
    "not_object_items": re.compile(r"must be a JSON object\. Problematic item:"),
    "triple_extraction_failures": re.compile(r"Triple extraction failed:"),
    "semantic_filter_items": re.compile(r"Post-validation dropped event_relation item:"),
}
PROCESSED = re.compile(r"Processed\s+(\d+)/(\d+)\s+chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Directory containing experiment/ and extract.log")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--pilot-chunks", type=int, default=109)
    parser.add_argument("--manual-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(run_root: Path, limit: int) -> list[dict]:
    extraction = run_root / "experiment" / "graph" / "kg_extraction"
    files = sorted(path for path in extraction.glob("*.json") if "hotpotqa_corpus" in path.name)
    if not files:
        raise FileNotFoundError(f"No extraction JSONL found in {extraction}")
    records: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt JSONL {path}:{line_number}") from exc
                if not isinstance(record, dict) or not {"id", "original_text"} <= record.keys():
                    raise ValueError(f"Invalid record {path}:{line_number}")
                records.append(record)
                if len(records) == limit:
                    return records
    return records


def parse_log(run_root: Path, limit: int) -> dict[int, Counter]:
    log_path = run_root / "extract.log"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    by_chunk: dict[int, Counter] = {}
    pending = Counter()
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            # A failed attempt may leave warnings without a durable record. Do not
            # attach those warnings to the first chunk of the next invocation.
            if "run_colab_v1.py" in line and "Running:" in line:
                pending.clear()
            for name, pattern in WARNING_PATTERNS.items():
                if pattern.search(line):
                    pending[name] += 1
            match = PROCESSED.search(line)
            if match:
                chunk = int(match.group(1))
                if chunk <= limit:
                    by_chunk.setdefault(chunk, Counter()).update(pending)
                pending.clear()
    return by_chunk


def stage_items(record: dict, stage: str) -> list:
    value = record.get(f"{stage}_dict", [])
    return value if isinstance(value, list) else []


def make_audit(records: list[dict], warnings: dict[int, Counter]) -> dict:
    per_chunk = [sum(len(stage_items(record, stage)) for stage in STAGES) for record in records]
    stage_distribution = {
        stage: sum(len(stage_items(record, stage)) for record in records)
        for stage in STAGES
    }
    warning_totals = Counter()
    for chunk in range(1, len(records) + 1):
        warning_totals.update(warnings.get(chunk, {}))
    for name in WARNING_PATTERNS:
        warning_totals.setdefault(name, 0)
    valid = sum(per_chunk)
    dropped_names = (
        "missing_field_items", "duplicate_items", "empty_value_items",
        "null_value_items", "invalid_array_items", "not_object_items",
        "semantic_filter_items",
    )
    dropped = sum(warning_totals[name] for name in dropped_names)
    candidates = valid + dropped
    empty = sum(value == 0 for value in per_chunk)
    return {
        "chunks": len(records),
        "valid_saved_items": valid,
        "estimated_raw_candidates": candidates,
        "dropped_candidates": dropped,
        "estimated_keep_rate": valid / candidates if candidates else 0.0,
        "warnings": dict(warning_totals),
        "estimated_rates": {
            name.removesuffix("_items") + "_rate": warning_totals[name] / candidates if candidates else 0.0
            for name in dropped_names
        },
        "empty_chunks": empty,
        "empty_chunk_rate": empty / len(records) if records else 0.0,
        "valid_items_per_chunk": {
            "mean": statistics.mean(per_chunk) if per_chunk else 0.0,
            "median": statistics.median(per_chunk) if per_chunk else 0.0,
            "min": min(per_chunk, default=0),
            "max": max(per_chunk, default=0),
        },
        "stage_valid_distribution": stage_distribution,
        "note": "Warning-derived rates come from extract.log; *_dict fields contain post-validation output.",
    }


def make_concentration(records: list[dict], warnings: dict[int, Counter]) -> dict:
    affected = []
    for chunk in range(1, len(records) + 1):
        counts = warnings.get(chunk, Counter())
        duplicates = counts["duplicate_items"]
        if duplicates:
            affected.append({
                "chunk": chunk,
                "duplicates": duplicates,
                "missing": counts["missing_field_items"],
                "empty": counts["empty_value_items"],
            })
    affected.sort(key=lambda item: (-item["duplicates"], item["chunk"]))
    values = [item["duplicates"] for item in affected]
    return {
        "parsed_chunks": len(records),
        "chunks_with_duplicates": len(affected),
        "duplicate_chunk_rate": len(affected) / len(records) if records else 0.0,
        "duplicates_total": sum(values),
        "duplicates_per_affected_chunk": {
            "mean": statistics.mean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "max": max(values, default=0),
        },
        "top_10_repetition_chunks": affected[:10],
    }


def make_manual_review(records: list[dict], warnings: dict[int, Counter], size: int, seed: int) -> list[dict]:
    high_count = min(size // 2, len(records))
    ranked = sorted(
        range(1, len(records) + 1),
        key=lambda chunk: (-warnings.get(chunk, Counter())["duplicate_items"], chunk),
    )
    high = [chunk for chunk in ranked if warnings.get(chunk, Counter())["duplicate_items"] > 0][:high_count]
    unaffected = [
        chunk for chunk in range(1, len(records) + 1)
        if warnings.get(chunk, Counter())["duplicate_items"] == 0 and chunk not in high
    ]
    random_sample = random.Random(seed).sample(unaffected, min(size - len(high), len(unaffected)))
    selected = [(chunk, "high_repetition") for chunk in high]
    selected += [(chunk, "random_unaffected") for chunk in random_sample]
    review = []
    for chunk, group in selected:
        record = records[chunk - 1]
        review.append({
            "chunk": chunk,
            "group": group,
            "duplicate_warnings": warnings.get(chunk, Counter())["duplicate_items"],
            "id": record["id"],
            "text": record["original_text"],
            **{stage: stage_items(record, stage) for stage in STAGES},
            "manual_labels": {
                "entity_relation": None,
                "event_entity": None,
                "event_relation": None,
                "overall": None,
                "notes": "",
            },
        })
    return review


def write_manual_markdown(path: Path, review: list[dict]) -> None:
    lines = [
        "# Targeted manual review",
        "",
        "Điền nhãn `correct`, `partial`, `incorrect` hoặc `empty` vào file JSON tương ứng.",
        "",
    ]
    for item in review:
        lines += [
            f"## Chunk {item['chunk']} — {item['group']}", "",
            f"Duplicate warnings: {item['duplicate_warnings']}", "",
            "### Text", "", item["text"], "",
        ]
        for stage in STAGES:
            lines += [f"### {stage}", "", "```json",
                      json.dumps(item[stage], ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_baseline_audit(root: Path) -> dict:
    candidates = (
        root / "pilot_audit_100" / "corrected_log_audit.json",
        root / "pilot_audit_109" / "corrected_log_audit.json",
    )
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    records = load_records(root, 109)
    return make_audit(records, parse_log(root, len(records)))


def comparison(baseline: dict, candidate: dict) -> dict:
    paths = {
        "keep_rate": ("estimated_keep_rate",),
        "duplicate_rate": ("estimated_rates", "duplicate_rate"),
        "missing_field_rate": ("estimated_rates", "missing_field_rate"),
        "empty_chunk_rate": ("empty_chunk_rate",),
        "valid_items_mean": ("valid_items_per_chunk", "mean"),
    }
    def get(data: dict, keys: tuple[str, ...]):
        for key in keys:
            data = data[key]
        return data
    return {
        name: {"baseline": get(baseline, keys), "candidate": get(candidate, keys),
               "delta": get(candidate, keys) - get(baseline, keys)}
        for name, keys in paths.items()
    }


def main() -> None:
    args = parse_args()
    if args.pilot_chunks < 1 or args.manual_size < 1:
        raise ValueError("pilot-chunks and manual-size must be positive")
    root = args.run_root.expanduser().resolve()
    records = load_records(root, args.pilot_chunks)
    if len(records) < args.pilot_chunks:
        raise RuntimeError(f"Pilot incomplete: found {len(records)}/{args.pilot_chunks} durable chunks")
    warnings = parse_log(root, len(records))
    audit = make_audit(records, warnings)
    concentration = make_concentration(records, warnings)
    review = make_manual_review(records, warnings, args.manual_size, args.seed)

    output = root / f"pilot_audit_{len(records)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "corrected_log_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "duplicate_concentration.json").write_text(
        json.dumps(concentration, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "targeted_manual_review_20.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manual_markdown(output / "targeted_manual_review_20.md", review)

    print("\nCORRECTED AUDIT")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nDUPLICATE CONCENTRATION")
    print(json.dumps(concentration, ensure_ascii=False, indent=2))
    if args.baseline_root:
        result = comparison(load_baseline_audit(args.baseline_root.expanduser().resolve()), audit)
        (output / "baseline_comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nBASELINE COMPARISON")
        print(f"{'metric':<24} {'baseline':>12} {'candidate':>12} {'delta':>12}")
        for name, values in result.items():
            print(f"{name:<24} {values['baseline']:>12.4f} {values['candidate']:>12.4f} {values['delta']:>+12.4f}")
    print(f"\nSaved reports: {output}")
    print(f"Manual review: {output / 'targeted_manual_review_20.md'}")


if __name__ == "__main__":
    main()
