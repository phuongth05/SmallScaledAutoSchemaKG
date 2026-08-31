from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_hotpotqa_benchmark import error_taxonomy, make_report
from run_hotpotqa_en import parse_args, prepared_protocol


def record(question_id, em, recall5, f1=None):
    return {
        "id": question_id,
        "metrics": {
            "em": em,
            "f1": em if f1 is None else f1,
            "support_recall@2": recall5,
            "support_recall@5": recall5,
        },
        "elapsed_seconds": 1,
    }


def test_default_protocol_is_seeded_100_question_validation_sample():
    args = parse_args([])
    assert prepared_protocol(args) == {
        "dataset_id": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "sampling": "random",
        "seed": 42,
        "max_questions_requested": 100,
        "context_mode": "all",
    }


def test_error_taxonomy_separates_retrieval_and_reader_failures():
    taxonomy = error_taxonomy({
        "correct": record("correct", 1, 1),
        "retrieval": record("retrieval", 0, 0.5),
        "reader": record("reader", 0, 1),
    })
    assert taxonomy["counts"] == {"correct": 1, "retrieval_failure": 1, "reader_failure": 1}


def test_report_writes_paper_metrics_and_paired_comparison(tmp_path):
    source = tmp_path / "graph" / "provenance"
    source.mkdir(parents=True)
    manifest = [
        {"id": "q1", "type": "bridge", "level": "hard"},
        {"id": "q2", "type": "comparison", "level": "hard"},
    ]
    (source / "qa_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    benchmark = tmp_path / "benchmark"
    summary = {
        "protocol": "test",
        "methods": {
            "dense": {"diagnostics": {}},
            "entity": {"diagnostics": {"filter_retry_calls": 0}},
        },
    }
    benchmark.mkdir()
    (benchmark / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for method, records in {
        "dense": [record("q1", 0, 0.5), record("q2", 1, 1)],
        "entity": [record("q1", 1, 1), record("q2", 1, 1)],
    }.items():
        folder = benchmark / "results" / method
        folder.mkdir(parents=True)
        for item in records:
            (folder / f"{item['id']}.json").write_text(json.dumps(item), encoding="utf-8")

    report = make_report(benchmark, tmp_path / "graph", repetitions=100)
    assert report["results"]["entity"]["metrics"]["em"] == 1
    assert report["paired_bootstrap_vs_baseline"]["entity"]["em"]["delta"] == 0.5
    assert (benchmark / "evaluation_100q.md").is_file()
    assert (benchmark / "evaluation_100q.json").is_file()
