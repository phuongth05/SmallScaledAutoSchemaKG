"""Run staged AutoSchemaKG experiments on HotpotQA-VI-1K final/.

prepare is CPU-only. extract/build/all require a running local Qwen server.
benchmark reuses the Vietnamese graph; it never accepts an English graph as a
substitute. Run benchmark again unchanged to resume per-question checkpoints.
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

from prepare_hotpotqa_vn import prepare

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "extract", "build", "package", "benchmark", "all"), default="prepare")
    parser.add_argument("--source-dir", type=Path, help="Path to Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/hotpotqa_vn"))
    parser.add_argument("--max-questions", type=int, help="Default all queries; the full corpus is always retained")
    parser.add_argument("--sampling", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Characters, not tokens")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-extraction-chunks", type=int,
                        help="Gracefully checkpoint extract after N new chunks; rerun unchanged to continue")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--top-passages", type=int, default=5,
                        help="Retrieved passage budget for benchmark; use 10 to report Recall@2/@5/@10")
    parser.add_argument("--model-revision", help="Tokenizer revision; server must use matching model weights")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--variants", nargs="+", choices=("dense", "entity", "entity_event", "full", "no_event"),
                        default=["dense", "entity", "entity_event", "full"])
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-filter-edges", action="store_true")
    parser.add_argument("--without-event-relations", action="store_true",
                        help="Ablation: keep entity/event nodes but skip event-event extraction")
    parser.add_argument("--without-events", action="store_true",
                        help="No-event ablation: extract Entity-Entity only; create no event nodes or edges")
    parser.add_argument("--allow-partial-build", action="store_true",
                        help="Pilot only: build/benchmark after partial extraction while retaining all corpus passages")
    return parser.parse_args()


def call_script(name, arguments):
    command = [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts" / name), *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    environment = dict(os.environ, PYTHONUTF8="1", USE_TF="0", USE_FLAX="0")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def construction_args(args, data_dir, graph_dir, phase):
    command = ["--data-dir", data_dir, "--filename-pattern", "hotpotqa_corpus",
            "--experiment-metadata", data_dir / "dataset_metadata.json",
            "--output-dir", graph_dir, "--phase", phase, "--language", "vi",
            "--model", args.model, "--base-url", args.base_url,
            "--chunk-size", args.chunk_size, "--max-new-tokens", args.max_new_tokens,
            "--repetition-penalty", args.repetition_penalty]
    if getattr(args, "without_event_relations", False):
        command.append("--without-event-relations")
    if getattr(args, "without_events", False):
        command.append("--without-events")
    if phase == "extract":
        command.append("--resume-extraction")
        if getattr(args, "max_extraction_chunks", None):
            command += ["--max-extraction-chunks", args.max_extraction_chunks]
    return command


def main():
    args = parse_args()
    work = args.work_dir.resolve()
    data, graph = work / "input", work / "graph"
    if args.phase in {"prepare", "all"}:
        if args.source_dir is None:
            raise ValueError("--source-dir is required for prepare/all")
        prepare(args.source_dir, data, args.max_questions, args.sampling, args.seed)
    if args.phase == "prepare":
        return
    metadata = json.loads((data / "dataset_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("language") != "vi" or metadata.get("retrieval_ground_truth") != "qrels_document_ids":
        raise ValueError("Expected Vietnamese inputs prepared by prepare_hotpotqa_vn.py")
    graph.mkdir(parents=True, exist_ok=True)
    provenance = graph / "provenance"
    provenance.mkdir(exist_ok=True)
    for name in ("hotpotqa_corpus.json", "qa_manifest.json", "dataset_metadata.json"):
        target = provenance / name
        if target.exists() and target.read_bytes() != (data / name).read_bytes():
            raise ValueError("Graph provenance differs from prepared inputs; use a NEW work directory")
        if not target.exists():
            shutil.copy2(data / name, target)
    if args.phase in {"extract", "build", "all"}:
        config = {"model": args.model, "model_revision": args.model_revision, "language": "vi", "chunk_size": args.chunk_size,
                  "max_new_tokens": args.max_new_tokens,
                  "repetition_penalty": args.repetition_penalty,
                  "include_events": not args.without_events,
                  "include_event_relations": not args.without_event_relations,
                  "corpus_sha256": hashlib.sha256((data / "hotpotqa_corpus.json").read_bytes()).hexdigest(),
                  "prompts_sha256": hashlib.sha256((ROOT / "atlas_rag/llm_generator/prompt/vietnamese.py").read_bytes()).hexdigest()}
        config_file = graph / "vn_construction_config.json"
        if config_file.exists():
            previous = json.loads(config_file.read_text(encoding="utf-8"))
            if previous != config:
                changed = {key: {"saved": previous.get(key), "requested": config.get(key)}
                           for key in sorted(set(previous) | set(config))
                           if previous.get(key) != config.get(key)}
                raise ValueError(
                    "Construction configuration changed; use a NEW work directory. "
                    f"Differences: {json.dumps(changed, ensure_ascii=False)}"
                )
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.phase in {"extract", "all"}:
        marker = graph / "vn_extraction_complete.json"
        if marker.exists():
            print("Extraction already completed; keeping saved output.", flush=True)
        else:
            if (graph / "kg_extraction").exists() and any((graph / "kg_extraction").iterdir()):
                print("Partial extraction found; validating JSONL checkpoints and resuming.", flush=True)
            call_script("run_colab_v1.py", construction_args(args, data, graph, "extract"))
            progress_file = graph / "extraction_progress.json"
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
            if progress.get("complete"):
                marker.write_text('{"complete": true}', encoding="utf-8")
            else:
                print(
                    f"Extraction checkpoint saved: {progress.get('completed_chunks')}/"
                    f"{progress.get('total_chunks')} chunks. Rerun phase extract unchanged to continue.",
                    flush=True,
                )
                return
    if args.phase in {"build", "all"}:
        if not (graph / "vn_extraction_complete.json").exists():
            if not args.allow_partial_build:
                raise RuntimeError("Complete the extract phase before build, or explicitly pass --allow-partial-build for a pilot")
            progress_file = graph / "extraction_progress.json"
            progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.is_file() else {}
            if not progress.get("completed_chunks"):
                raise RuntimeError("Partial build requires at least one durable extracted chunk")
            print("PARTIAL PILOT BUILD: all corpus passages are retained; only extracted chunks have KG edges.", flush=True)
        marker = graph / "vn_build_complete.json"
        if not marker.exists():
            call_script("run_colab_v1.py", construction_args(args, data, graph, "build"))
            if not (graph / "kg_graphml/hotpotqa_corpus_graph.graphml").is_file():
                raise FileNotFoundError("Build returned without the expected GraphML; not marking complete")
            marker.write_text('{"complete": true}', encoding="utf-8")
        else:
            print("Graph build already completed.", flush=True)
    if args.phase in {"package", "benchmark", "all"}:
        if not (graph / "vn_build_complete.json").exists():
            raise RuntimeError("Complete the build phase before packaging/benchmark")
        if not (graph / "kg_graphml/hotpotqa_corpus_graph.graphml").is_file():
            raise FileNotFoundError("Vietnamese GraphML is missing")
    if args.phase in {"package", "all"}:
        archive = shutil.make_archive(str(work / "hotpotqa_vn_graph"), "zip", graph)
        print(f"Graph/provenance archive: {archive}", flush=True)
    if args.phase in {"benchmark", "all"}:
        command = [graph, "--output-dir", work / "benchmark", "--language", "vi",
                   "--embedding-model", args.embedding_model, "--embedding-device", "cpu",
                   "--model", args.model, "--base-url", args.base_url,
                   "--context-length", args.context_length, "--top-passages", args.top_passages,
                   "--variants", *args.variants]
        for name in ("retrieval_only", "no_filter_edges"):
            if getattr(args, name):
                command.append("--" + name.replace("_", "-"))
        for name in ("model_revision", "embedding_revision"):
            if getattr(args, name):
                command += ["--" + name.replace("_", "-"), getattr(args, name)]
        call_script("run_hotpotqa_benchmark.py", command)


if __name__ == "__main__":
    main()
