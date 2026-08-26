"""Inspect an AutoSchemaKG HotpotQA ZIP and optionally run local-LLM QA.

The ZIP must contain the GraphML graph plus the provenance files produced by
the HotpotQA Colab construction workflow. QA uses only graph triples; gold
answers are used after generation for EM/F1 scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/hotpotqa_from_zip"))
    parser.add_argument("--output-zip", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to use unsafe path: {resolved}")
    return resolved


def extract_zip(zip_path: Path, work_dir: Path, overwrite: bool) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if work_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{work_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    root = work_dir.resolve()
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (work_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(work_dir)


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename!r} in {root}, found {matches}")
    return matches[0]


def load_artifacts(work_dir: Path) -> tuple[Any, list[dict], list[dict], Path]:
    import networkx as nx

    graph_path = find_one(work_dir, "hotpotqa_corpus_graph.graphml")
    manifest_path = find_one(work_dir, "qa_manifest.json")
    corpus_path = find_one(work_dir, "hotpotqa_corpus.json")
    graph = nx.read_graphml(graph_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not manifest or not corpus:
        raise ValueError("HotpotQA manifest or corpus is empty")
    return graph, manifest, corpus, graph_path


def graph_summary(graph: Any) -> dict[str, Any]:
    import networkx as nx

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
        "node_types": dict(Counter(d.get("type", "unknown") for _, d in graph.nodes(data=True))),
        "top_relations": Counter(
            d.get("relation", "unknown") for _, _, d in graph.edges(data=True)
        ).most_common(20),
    }


def save_visualization(graph: Any, output_path: Path, max_nodes: int = 80) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    eligible = [n for n, d in graph.nodes(data=True) if d.get("type") != "passage"]
    important = sorted(eligible, key=lambda node: graph.degree(node), reverse=True)[:max_nodes]
    subgraph = graph.subgraph(important).copy()
    colors = {
        "entity": "#4C78A8",
        "event": "#F58518",
        "concept": "#54A24B",
        "triple": "#B279A2",
        "unknown": "#9D9D9D",
    }
    node_colors = [
        colors.get(subgraph.nodes[n].get("type", "unknown"), colors["unknown"])
        for n in subgraph.nodes
    ]
    labels = {n: str(subgraph.nodes[n].get("id", n))[:25] for n in subgraph.nodes}
    figure = plt.figure(figsize=(20, 14))
    positions = nx.spring_layout(subgraph, seed=42, k=1.2, iterations=100)
    nx.draw_networkx_nodes(subgraph, positions, node_color=node_colors, node_size=450)
    nx.draw_networkx_edges(subgraph, positions, alpha=0.25, arrows=True)
    nx.draw_networkx_labels(subgraph, positions, labels=labels, font_size=7)
    plt.title("AutoSchemaKG HotpotQA")
    plt.axis("off")
    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_document_to_passage_map(graph: Any, corpus: list[dict]) -> dict[str, str]:
    """Map provenance IDs to GraphML passage IDs through exact passage text.

    AutoSchemaKG hashes passage IDs during CSV conversion, so provenance IDs
    cannot be compared directly with GraphML ``file_id`` values.
    """

    text_to_passage = {
        str(data.get("id", "")): str(node)
        for node, data in graph.nodes(data=True)
        if data.get("type") == "passage"
    }
    mapping = {
        str(document["id"]): text_to_passage.get(str(document["text"]), "")
        for document in corpus
    }
    missing = [document_id for document_id, passage_id in mapping.items() if not passage_id]
    if missing:
        raise ValueError(f"Could not map {len(missing)} corpus documents into GraphML")
    return mapping


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def retrieve_triples(
    graph: Any,
    question: str,
    document_ids: list[str],
    document_to_passage: dict[str, str],
    top_k: int,
) -> list[str]:
    passage_ids = {document_to_passage[document_id] for document_id in document_ids}
    question_tokens = tokenize(question)
    ranked: list[tuple[int, int, str]] = []

    for source, target, edge_data in graph.edges(data=True):
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        relation = str(edge_data.get("relation", "related to"))
        if source_data.get("type") in {"passage", "concept"}:
            continue
        if target_data.get("type") in {"passage", "concept"}:
            continue
        if relation == "mention in":
            continue
        file_ids = {
            value.strip()
            for value in (
                f"{source_data.get('file_id', '')},{target_data.get('file_id', '')}"
            ).split(",")
            if value.strip()
        }
        if passage_ids.isdisjoint(file_ids):
            continue
        triple = (
            f"({source_data.get('id', source)}, {relation}, "
            f"{target_data.get('id', target)})"
        )
        overlap = len(question_tokens & tokenize(triple))
        degree = graph.degree(source) + graph.degree(target)
        ranked.append((overlap, degree, triple))

    ranked.sort(reverse=True)
    triples = [triple for _, _, triple in ranked[:top_k]]
    if not triples:
        raise RuntimeError("Retriever returned no graph triples; refusing answer-only evaluation")
    return triples


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def calculate_scores(prediction: str, reference: str) -> tuple[int, float]:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    exact_match = int(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    same = sum(common.values())
    if same == 0:
        return exact_match, 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(reference_tokens)
    return exact_match, 2 * precision * recall / (precision + recall)


def run_qa(
    graph: Any,
    manifest: list[dict],
    corpus: list[dict],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import requests
    from openai import OpenAI

    model_response = requests.get(f"{args.base_url.rstrip('/')}/models", timeout=10)
    model_response.raise_for_status()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    document_to_passage = build_document_to_passage_map(graph, corpus)
    results = []
    started = time.monotonic()

    for index, sample in enumerate(manifest, start=1):
        triples = retrieve_triples(
            graph,
            str(sample["question"]),
            [str(value) for value in sample["document_ids"]],
            document_to_passage,
            args.top_k,
        )
        context = "\n".join(triples)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using the supplied knowledge graph triples. Return only "
                        "the shortest exact answer without explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {sample['question']}\n\nKnowledge graph:\n{context}"
                        "\n\nShort answer:"
                    ),
                },
            ],
            temperature=0,
            max_tokens=args.max_answer_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        prediction = (response.choices[0].message.content or "").strip()
        prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL).strip()
        exact_match, f1 = calculate_scores(prediction, str(sample["answer"]))
        result = {
            "id": sample["id"],
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "prediction": prediction,
            "exact_match": exact_match,
            "f1": f1,
            "retrieved_triple_count": len(triples),
            "retrieved_triples": triples,
        }
        results.append(result)
        print(
            f"[{index}/{len(manifest)}] EM={exact_match} F1={f1:.4f} "
            f"triples={len(triples)} prediction={prediction!r}",
            flush=True,
        )

    return {
        "model": args.model,
        "questions": len(results),
        "mean_exact_match": sum(item["exact_match"] for item in results) / len(results),
        "mean_f1": sum(item["f1"] for item in results) / len(results),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }


def package(work_dir: Path, output_zip: Path) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    base = output_zip.with_suffix("")
    archive = shutil.make_archive(str(base), "zip", work_dir)
    return Path(archive)


def main() -> None:
    args = parse_args()
    args.zip_path = args.zip_path.expanduser().resolve()
    args.work_dir = safe_path(args.work_dir)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    extract_zip(args.zip_path, args.work_dir, args.overwrite)
    graph, manifest, corpus, graph_path = load_artifacts(args.work_dir)
    summary = graph_summary(graph)
    print(json.dumps(summary, indent=2))
    print(f"GraphML: {graph_path}")
    image_path = args.work_dir / "hotpotqa_graph_overview.png"
    save_visualization(graph, image_path)
    print(f"Visualization: {image_path}")

    if not args.visualize_only:
        evaluation = run_qa(graph, manifest, corpus, args)
        result_path = args.work_dir / "hotpotqa_kg_qa_results.json"
        result_path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in evaluation.items() if key != "results"}, indent=2))
        print(f"QA results: {result_path}")

    output_zip = args.output_zip
    if output_zip is None:
        output_zip = args.zip_path.with_name(f"{args.zip_path.stem}_reanalyzed.zip")
    archive = package(args.work_dir, safe_path(output_zip))
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
