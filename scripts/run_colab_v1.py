"""Run the small-scale AutoSchemaKG pipeline against a local OpenAI API.

The default endpoint is a local vLLM server hosting Qwen3.5-2B. The script is
also compatible with any OpenAI-compatible local server exposing the same
model name.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATA_DIR = REPO_ROOT / "example" / "example_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--language", choices=("en", "vi"), default="en", help="Concept-induction language; input metadata.lang selects extraction prompts")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--filename-pattern", default="v1_smoke")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "colab_v1")
    parser.add_argument(
        "--phase",
        choices=("extract", "build", "full"),
        default="full",
        help="extract: LLM extraction only; build: convert existing extraction; full: both",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="vLLM repetition penalty used by all extraction stages.",
    )
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument(
        "--experiment-metadata",
        type=Path,
        help="Optional JSON metadata copied into run_summary.json for provenance.",
    )
    parser.add_argument("--without-concepts", action="store_true")
    parser.add_argument(
        "--without-event-relations",
        action="store_true",
        help="Extract entity relations and event-entity edges, but skip event-event relations.",
    )
    parser.add_argument(
        "--resume-extraction",
        action="store_true",
        help="Resume after valid JSONL records already saved in kg_extraction.",
    )
    parser.add_argument(
        "--max-extraction-chunks",
        type=int,
        help="Gracefully checkpoint after at most this many new chunks; default runs all remaining chunks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the selected output directory before starting.",
    )
    return parser.parse_args()


def inspect_extraction_progress(output_dir: Path, filename_pattern: str) -> dict:
    """Count durable extraction records and reject corrupt checkpoints."""
    extraction_dir = output_dir / "kg_extraction"
    files = sorted(
        path for path in extraction_dir.glob("*.json")
        if filename_pattern in path.name
    ) if extraction_dir.is_dir() else []
    records = 0
    file_counts = {}
    for path in files:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corrupt extraction checkpoint {path}:{line_number}; "
                        "preserve the file and inspect its final line before resuming"
                    ) from exc
                if not isinstance(item, dict) or not {"id", "original_text"} <= set(item):
                    raise ValueError(f"Invalid extraction checkpoint record {path}:{line_number}")
                count += 1
        file_counts[path.name] = count
        records += count
    return {"completed_chunks": records, "files": file_counts}


def write_extraction_progress(output_dir: Path, progress: dict) -> None:
    payload = dict(progress, updated_at_utc=datetime.now(timezone.utc).isoformat())
    target = output_dir / "extraction_progress.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)


def ensure_safe_output(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir == Path(output_dir.anchor) or len(output_dir.parts) < 3:
        raise ValueError(f"Refusing to use unsafe output directory: {output_dir}")
    return output_dir


def make_extractor(args: argparse.Namespace) -> object:
    from openai import OpenAI

    from atlas_rag.kg_construction.triple_config import ProcessingConfig
    from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
    from atlas_rag.llm_generator import GenerationConfig, LLMGenerator

    generation = GenerationConfig(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=args.repetition_penalty,
        do_sample=False,
        seed=42,
        chat_template_kwargs={"enable_thinking": False},
    )
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800.0)
    generator = LLMGenerator(
        client=client,
        model_name=args.model,
        backend="vllm",
        max_workers=1,
        default_config=generation,
    )
    config = ProcessingConfig(
        model_path=args.model,
        data_directory=str(args.data_dir.resolve()),
        filename_pattern=args.filename_pattern,
        batch_size_triple=1,
        batch_size_concept=8,
        output_directory=str(args.output_dir),
        max_new_tokens=args.max_new_tokens,
        max_workers=1,
        remove_doc_spaces=True,
        include_concept=not args.without_concepts,
        include_event_relations=not args.without_event_relations,
        record=True,
        chunk_size=args.chunk_size,
        allow_empty=False,
        resume_from=args.resume_from,
        max_chunks_per_run=args.max_extraction_chunks,
    )
    return KnowledgeGraphExtractor(model=generator, config=config)


def build_summary(args: argparse.Namespace) -> dict:
    import networkx as nx

    graph_path = args.output_dir / "kg_graphml" / f"{args.filename_pattern}_graph.graphml"
    summary = {
        "model": args.model,
        "language": args.language,
        "data": str(args.data_dir / f"{args.filename_pattern}.json"),
        "output_directory": str(args.output_dir),
        "include_concepts": not args.without_concepts,
        "include_event_relations": not args.without_event_relations,
        "graphml": str(graph_path),
    }
    if args.experiment_metadata:
        metadata_path = args.experiment_metadata.expanduser().resolve()
        summary["experiment_metadata_file"] = str(metadata_path)
        summary["experiment"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    if graph_path.exists():
        graph = nx.read_graphml(graph_path)
        node_types: dict[str, int] = {}
        for _, attributes in graph.nodes(data=True):
            node_type = str(attributes.get("type", "unknown"))
            node_types[node_type] = node_types.get(node_type, 0) + 1
        summary.update(
            {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "node_types": node_types,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = ensure_safe_output(args.output_dir)
    input_files = list(args.data_dir.glob(f"{args.filename_pattern}*.json*"))
    if not input_files:
        raise FileNotFoundError(
            f"No input matching {args.filename_pattern}*.json* in {args.data_dir}"
        )
    if args.overwrite and args.resume_extraction:
        raise ValueError("--overwrite and --resume-extraction are mutually exclusive")
    if args.max_extraction_chunks is not None and args.max_extraction_chunks < 1:
        raise ValueError("--max-extraction-chunks must be positive")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress = inspect_extraction_progress(args.output_dir, args.filename_pattern)
    if progress["completed_chunks"] and not args.resume_extraction and args.phase in {"extract", "full"}:
        raise FileExistsError(
            "Partial extraction exists. Pass --resume-extraction to continue without duplicating chunks."
        )
    args.resume_from = progress["completed_chunks"] if args.resume_extraction else 0
    if args.resume_from:
        print(
            f"Resuming extraction after {args.resume_from} durable chunks "
            f"from {len(progress['files'])} file(s).",
            flush=True,
        )
    write_extraction_progress(args.output_dir, progress)

    extractor = make_extractor(args)
    if args.phase in {"extract", "full"}:
        extraction_result = extractor.run_extraction()
        progress = inspect_extraction_progress(args.output_dir, args.filename_pattern)
        progress.update(extraction_result or {})
        write_extraction_progress(args.output_dir, progress)
    if args.phase in {"build", "full"}:
        extractor.convert_json_to_csv()
        if not args.without_concepts:
            extractor.generate_concept_csv_temp(language=args.language)
            extractor.create_concept_csv()
        extractor.convert_to_graphml()

    summary = build_summary(args)
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
