"""Stream a reproducible HotpotQA slice and prepare AutoSchemaKG inputs.

The knowledge-graph corpus contains Wikipedia passages only. Questions,
answers, and sentence-level supporting-fact labels are written to a separate
manifest so gold answers are never injected into the graph construction input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "hotpotqa/hotpot_qa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("distractor", "fullwiki"), default="distractor")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--loader",
        choices=("api", "datasets"),
        default="api",
        help="api uses the fast Dataset Viewer rows endpoint; datasets uses streaming=True.",
    )
    parser.add_argument(
        "--context-mode",
        choices=("all", "supporting"),
        default="all",
        help="all keeps distractors; supporting keeps only gold supporting documents.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa_v1"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_supporting_pairs(value: Any) -> list[list[Any]]:
    if isinstance(value, dict):
        return [list(pair) for pair in zip(value.get("title", []), value.get("sent_id", []))]
    return [list(pair) for pair in (value or [])]


def as_context_pairs(value: Any) -> list[tuple[str, list[str]]]:
    if isinstance(value, dict):
        return [
            (str(title), [str(sentence) for sentence in sentences])
            for title, sentences in zip(value.get("title", []), value.get("sentences", []))
        ]
    return [
        (str(title), [str(sentence) for sentence in sentences])
        for title, sentences in (value or [])
    ]


def stable_document_id(title: str, text: str) -> str:
    digest = hashlib.sha256(f"{title}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"hotpotqa-{digest}"


def load_examples_with_datasets(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        args.config,
        split=args.split,
        streaming=True,
    )
    import itertools

    return itertools.islice(dataset, args.start_index, args.start_index + args.max_questions)


def load_examples_with_api(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    import requests

    endpoint = "https://datasets-server.huggingface.co/rows"
    remaining = args.max_questions
    offset = args.start_index
    while remaining:
        length = min(remaining, 100)
        print(f"Requesting HotpotQA rows {offset}-{offset + length - 1} ...", flush=True)
        response = requests.get(
            endpoint,
            params={
                "dataset": DATASET_ID,
                "config": args.config,
                "split": args.split,
                "offset": offset,
                "length": length,
            },
            timeout=120,
        )
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item["row"]
        count = len(rows)
        remaining -= count
        offset += count
        print(f"Received {count} rows.", flush=True)
        if count < length:
            break


def load_examples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.loader == "api":
        return load_examples_with_api(args)
    return load_examples_with_datasets(args)


def validate_args(args: argparse.Namespace) -> Path:
    if args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.config == "distractor" and args.split == "test":
        raise ValueError("The distractor configuration has no test split")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == Path(output_dir.anchor) or len(output_dir.parts) < 3:
        raise ValueError(f"Refusing to use unsafe output directory: {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = validate_args(args)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents: dict[str, dict[str, Any]] = {}
    qa_manifest: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()

    for example in load_examples(args):
        question_id = str(example.get("id", example.get("_id", "")))
        supporting_facts = as_supporting_pairs(example.get("supporting_facts"))
        supporting_titles = {str(title) for title, _ in supporting_facts}
        document_ids: list[str] = []

        for title, sentences in as_context_pairs(example.get("context")):
            is_supporting = title in supporting_titles
            if args.context_mode == "supporting" and not is_supporting:
                continue
            passage = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
            text = f"{title}. {passage}"
            document_id = stable_document_id(title, text)
            document_ids.append(document_id)
            if document_id not in documents:
                documents[document_id] = {
                    "id": document_id,
                    "text": text,
                    "metadata": {
                        "lang": "en",
                        "source": DATASET_ID,
                        "title": title,
                        "question_ids": [question_id],
                        "is_supporting": is_supporting,
                    },
                }
            else:
                metadata = documents[document_id]["metadata"]
                if question_id not in metadata["question_ids"]:
                    metadata["question_ids"].append(question_id)
                metadata["is_supporting"] = metadata["is_supporting"] or is_supporting

        question_type = str(example.get("type", "unknown"))
        level = str(example.get("level", "unknown"))
        type_counts[question_type] += 1
        level_counts[level] += 1
        qa_manifest.append(
            {
                "id": question_id,
                "question": example.get("question"),
                "answer": example.get("answer"),
                "type": question_type,
                "level": level,
                "supporting_facts": supporting_facts,
                "document_ids": document_ids,
            }
        )

    if not qa_manifest:
        raise RuntimeError("No HotpotQA examples were received")
    if not documents:
        raise RuntimeError("No context documents were produced")

    corpus_path = output_dir / "hotpotqa_corpus.json"
    manifest_path = output_dir / "qa_manifest.json"
    metadata_path = output_dir / "dataset_metadata.json"
    corpus_path.write_text(
        json.dumps(list(documents.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(qa_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset_id": DATASET_ID,
        "config": args.config,
        "split": args.split,
        "access_method": "dataset-viewer-api" if args.loader == "api" else "datasets-streaming",
        "streaming": args.loader == "datasets",
        "start_index": args.start_index,
        "max_questions_requested": args.max_questions,
        "questions_written": len(qa_manifest),
        "unique_documents_written": len(documents),
        "context_mode": args.context_mode,
        "question_type_counts": dict(type_counts),
        "difficulty_counts": dict(level_counts),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_file": corpus_path.name,
        "qa_manifest_file": manifest_path.name,
        "license": "CC BY-SA 4.0",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Prepared data in {output_dir}")


if __name__ == "__main__":
    main()
