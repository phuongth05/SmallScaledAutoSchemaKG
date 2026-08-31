"""Run a reproducible staged English HotpotQA experiment.

The default protocol is a seeded 100-question sample from
hotpotqa/hotpot_qa distractor/validation.  Every durable artifact lives below
``--work-dir``.  Extraction and QA are resumable; changing protocol or model
settings requires a new work directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "extract", "build", "package", "benchmark", "report", "all"),
        default="prepare",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/hotpotqa_en_100q"))
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--sampling", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", choices=("distractor", "fullwiki"), default="distractor")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--loader", choices=("api", "datasets"), default="api")
    parser.add_argument("--context-mode", choices=("all", "supporting"), default="all")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--model-revision")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Characters, not tokens")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--max-extraction-chunks",
        type=int,
        help="Checkpoint after N new chunks; rerun the identical extract command to continue",
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("dense", "entity", "entity_event", "full"),
        default=["dense", "entity", "entity_event", "full"],
    )
    parser.add_argument("--top-edges", type=int, default=30)
    parser.add_argument("--top-passages", type=int, default=5)
    parser.add_argument("--ppr-alpha", type=float, default=0.9)
    parser.add_argument("--passage-weight", type=float, default=0.9)
    parser.add_argument("--filter-max-attempts", type=int, default=2)
    parser.add_argument("--filter-failure-policy", choices=("error", "dense"), default="error")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-filter-edges", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args(argv)


def call_script(name: str, arguments) -> None:
    command = [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts" / name), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    environment = dict(os.environ, PYTHONUTF8="1", USE_TF="0", USE_FLAX="0")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_protocol(args) -> dict:
    return {
        "dataset_id": "hotpotqa/hotpot_qa",
        "config": args.config,
        "split": args.split,
        "sampling": args.sampling,
        "seed": args.seed if args.sampling == "random" else None,
        "max_questions_requested": args.max_questions,
        "context_mode": args.context_mode,
    }


def validate_prepared(data: Path, args) -> dict:
    required = [data / name for name in ("hotpotqa_corpus.json", "qa_manifest.json", "dataset_metadata.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared input is incomplete: {missing}")
    metadata = json.loads((data / "dataset_metadata.json").read_text(encoding="utf-8"))
    expected = prepared_protocol(args)
    different = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if different:
        raise ValueError(f"Prepared protocol differs from requested settings: {different}. Use a NEW --work-dir")
    if metadata.get("questions_written") != args.max_questions:
        raise ValueError("Prepared question count is incomplete")
    return metadata


def copy_provenance(data: Path, graph: Path) -> None:
    provenance = graph / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in ("hotpotqa_corpus.json", "qa_manifest.json", "dataset_metadata.json"):
        source, target = data / name, provenance / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Saved graph provenance differs from prepared inputs; use a NEW work directory")
        if not target.exists():
            shutil.copy2(source, target)


def construction_args(args, data: Path, graph: Path, phase: str):
    command = [
        "--data-dir", data,
        "--filename-pattern", "hotpotqa_corpus",
        "--experiment-metadata", data / "dataset_metadata.json",
        "--output-dir", graph,
        "--phase", phase,
        "--language", "en",
        "--model", args.model,
        "--base-url", args.base_url,
        "--chunk-size", args.chunk_size,
        "--max-new-tokens", args.max_new_tokens,
    ]
    if phase == "extract":
        command.append("--resume-extraction")
        if args.max_extraction_chunks:
            command += ["--max-extraction-chunks", args.max_extraction_chunks]
    return command


def main(argv=None):
    args = parse_args(argv)
    if args.max_questions < 1:
        raise ValueError("--max-questions must be positive")
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    work = args.work_dir.expanduser().resolve()
    if work == Path(work.anchor) or len(work.parts) < 3:
        raise ValueError(f"Unsafe work directory: {work}")
    data, graph, benchmark = work / "input", work / "graph", work / "benchmark"

    if args.phase in {"prepare", "all"}:
        if (data / "dataset_metadata.json").exists():
            validate_prepared(data, args)
            print("Prepared input already exists and matches the protocol.", flush=True)
        else:
            command = [
                "--config", args.config, "--split", args.split,
                "--max-questions", args.max_questions, "--sampling", args.sampling,
                "--seed", args.seed, "--loader", args.loader,
                "--context-mode", args.context_mode, "--output-dir", data,
            ]
            call_script("prepare_hotpotqa.py", command)
    if args.phase == "prepare":
        return

    metadata = validate_prepared(data, args)
    graph.mkdir(parents=True, exist_ok=True)
    copy_provenance(data, graph)

    if args.phase in {"extract", "build", "all"}:
        config = {
            "model": args.model,
            "model_revision": args.model_revision,
            "language": "en",
            "chunk_size_characters": args.chunk_size,
            "max_new_tokens": args.max_new_tokens,
            "corpus_sha256": sha256(data / "hotpotqa_corpus.json"),
            "questions": metadata["questions_written"],
        }
        config_file = graph / "construction_config.json"
        if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("Construction configuration changed; use a NEW work directory")
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.phase in {"extract", "all"}:
        marker = graph / "extraction_complete.json"
        if marker.exists():
            print("Extraction already complete.", flush=True)
        else:
            call_script("run_colab_v1.py", construction_args(args, data, graph, "extract"))
            progress = json.loads((graph / "extraction_progress.json").read_text(encoding="utf-8"))
            if progress.get("complete"):
                marker.write_text('{"complete": true}', encoding="utf-8")
            else:
                print(
                    f"Extraction checkpoint: {progress.get('completed_chunks')}/{progress.get('total_chunks')}. "
                    "Rerun the identical extract command to continue.",
                    flush=True,
                )
                return

    if args.phase in {"build", "all"}:
        if not (graph / "extraction_complete.json").exists():
            raise RuntimeError("Complete extraction before build")
        marker = graph / "build_complete.json"
        graphml = graph / "kg_graphml" / "hotpotqa_corpus_graph.graphml"
        if not marker.exists():
            call_script("run_colab_v1.py", construction_args(args, data, graph, "build"))
            if not graphml.is_file():
                raise FileNotFoundError("Build finished without hotpotqa_corpus_graph.graphml")
            marker.write_text('{"complete": true}', encoding="utf-8")
        else:
            print("Graph build already complete.", flush=True)

    if args.phase in {"package", "benchmark", "report", "all"}:
        if not (graph / "build_complete.json").exists():
            raise RuntimeError("Complete graph build before package/benchmark/report")
        if not (graph / "kg_graphml" / "hotpotqa_corpus_graph.graphml").is_file():
            raise FileNotFoundError("GraphML is missing")

    if args.phase in {"package", "all"}:
        archive = shutil.make_archive(str(work / "construction"), "zip", graph)
        print("Construction archive:", archive, flush=True)

    if args.phase in {"benchmark", "all"}:
        command = [
            graph,
            "--output-dir", benchmark,
            "--language", "en",
            "--embedding-model", args.embedding_model,
            "--embedding-device", "cpu",
            "--model", args.model,
            "--base-url", args.base_url,
            "--context-length", args.context_length,
            "--variants", *args.variants,
            "--top-edges", args.top_edges,
            "--top-passages", args.top_passages,
            "--ppr-alpha", args.ppr_alpha,
            "--passage-weight", args.passage_weight,
            "--filter-max-attempts", args.filter_max_attempts,
            "--filter-failure-policy", args.filter_failure_policy,
        ]
        if args.retrieval_only:
            command.append("--retrieval-only")
        if args.no_filter_edges:
            command.append("--no-filter-edges")
        if args.model_revision:
            command += ["--model-revision", args.model_revision]
        if args.embedding_revision:
            command += ["--embedding-revision", args.embedding_revision]
        call_script("run_hotpotqa_benchmark.py", command)

    if args.phase in {"report", "all"}:
        call_script(
            "report_hotpotqa_benchmark.py",
            [benchmark, "--source", graph, "--baseline", "dense", "--bootstrap", args.bootstrap],
        )


if __name__ == "__main__":
    main()
