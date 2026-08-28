"""Convert Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final to KG inputs.

Default: keep the entire corpus, even for a small query subset. Gold qrels are
evaluation labels, not a document-selection filter. No translation is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

DATASET_URL = "https://github.com/chichic21039/Prepare-data-HotpotQA-VN"
FINAL_PATH = Path("data/hotpotqa_vi_1k/final")


def read_jsonl(path):
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                records.append(value)
    ids = [str(r["id"]) for r in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate IDs: {path}")
    return records


def load_final(source_dir):
    queries = read_jsonl(source_dir / "queries.jsonl")
    corpus = read_jsonl(source_dir / "corpus.jsonl")
    query_ids = {str(q["id"]) for q in queries}
    docs = {str(d["id"]): d for d in corpus}
    for query in queries:
        for key in ("question_vi", "answer_vi"):
            if not isinstance(query.get(key), str) or not query[key].strip():
                raise ValueError(f"{query['id']}: empty {key}; refusing English fallback")
    for document in corpus:
        if document.get("language") != "vi" or not str(document.get("text", "")).strip():
            raise ValueError(f"Invalid Vietnamese document {document['id']}")
        if not str(document.get("title", "")).strip():
            raise ValueError(f"Document {document['id']} has no title")
    qrels = defaultdict(set)
    pairs = set()
    with (source_dir / "qrels.tsv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"query_id", "corpus_id", "score"} <= set(reader.fieldnames or []):
            raise ValueError("qrels.tsv must contain query_id, corpus_id, score")
        for row in reader:
            qid, did = row["query_id"], row["corpus_id"]
            if qid not in query_ids or did not in docs:
                raise ValueError(f"Dangling qrel: {qid} -> {did}")
            if (qid, did) in pairs:
                raise ValueError(f"Duplicate qrel: {qid} -> {did}")
            pairs.add((qid, did))
            if float(row["score"]) > 0:
                qrels[qid].add(did)
    if set(qrels) != query_ids:
        raise ValueError("Every query must have at least one positive qrel")
    return queries, corpus, qrels


def prepare(source_dir, output_dir, max_questions=None, sampling="random", seed=42):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Set --source-dir to the dataset's {FINAL_PATH}: {source_dir}")
    queries, corpus, qrels = load_final(source_dir)
    count = len(queries) if max_questions is None else max_questions
    if not 1 <= count <= len(queries):
        raise ValueError(f"--max-questions must be between 1 and {len(queries)}")
    indices = (random.Random(seed).sample(range(len(queries)), count)
               if sampling == "random" else list(range(count)))
    selected = [queries[i] for i in indices]
    # Keep IDs stable and preserve ALL documents. No gold-dependent corpus pruning.
    documents = [{"id": str(d["id"]), "text": f"{d['title']}. {d['text']}",
                  "metadata": {"lang": "vi", "title": d["title"], "source": DATASET_URL,
                               "source_document_id": str(d["id"])}} for d in corpus]
    manifest = [{"id": str(q["id"]), "question": q["question_vi"], "answer": q["answer_vi"],
                 "language": "vi", "type": q.get("type", "unknown"), "level": q.get("level", "unknown"),
                 "document_ids": [],  # no per-query candidate list exists in final/
                 "gold_document_ids": sorted(qrels[str(q["id"])]),
                 "supporting_facts": [[f["title"], f["sent_id"]] for f in q.get("supporting_facts", [])],
                 "supporting_facts_language": "original_hotpotqa_titles_and_sentence_indices"}
                for q in selected]
    files = ["queries.jsonl", "corpus.jsonl", "qrels.tsv"]
    if (source_dir / "source_qa_issues.csv").exists():
        files.append("source_qa_issues.csv")
    sha256 = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in files}
    try:
        result = subprocess.run(["git", "-C", str(source_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
        revision = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    issues = []
    if "source_qa_issues.csv" in files:
        with (source_dir / "source_qa_issues.csv").open(encoding="utf-8-sig", newline="") as stream:
            issues = [r["id"] for r in csv.DictReader(stream)]
    metadata = {"dataset_id": "HotpotQA-VI-1K", "source_repository": DATASET_URL,
        "source_commit": revision, "source_files_sha256": sha256, "language": "vi",
        "split": "final", "context_mode": "all", "retrieval_ground_truth": "qrels_document_ids",
        "corpus_scope": "all_final_documents", "source_questions": len(queries),
        "questions_written": len(manifest), "unique_documents_written": len(documents),
        "sampling": sampling, "seed": seed if sampling == "random" else None,
        "selected_row_indices": indices, "selected_question_ids": [q["id"] for q in manifest],
        "positive_qrels_selected": sum(len(q["gold_document_ids"]) for q in manifest),
        "question_type_counts": dict(Counter(q["type"] for q in manifest)),
        "source_qa_issue_ids": issues, "source_qa_issues_policy": "kept; no automatic correction or exclusion",
        "corpus_file": "hotpotqa_corpus.json", "qa_manifest_file": "qa_manifest.json",
        "license_note": "See source repo and upstream datasets; no new license is asserted."}
    expected = {"hotpotqa_corpus.json": documents, "qa_manifest.json": manifest, "dataset_metadata.json": metadata}
    if output_dir.exists() and any(output_dir.iterdir()):
        if all((output_dir / name).is_file() and json.loads((output_dir / name).read_text(encoding="utf-8")) == content
               for name, content in expected.items()):
            print(f"Prepared inputs already match: {output_dir}", flush=True)
            return metadata
        raise FileExistsError(f"Use a NEW output directory; refusing to overwrite different inputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in expected.items():
        (output_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k not in {"selected_row_indices", "selected_question_ids"}},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"Prepared {len(manifest)} queries, ALL {len(documents)} documents. No LLM/GPU used.", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa_vn"))
    parser.add_argument("--max-questions", type=int, help="Default: all 1,000 queries; corpus is never reduced")
    parser.add_argument("--sampling", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.max_questions, args.sampling, args.seed)


if __name__ == "__main__":
    main()
