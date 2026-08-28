"""Shared, testable helpers for the paper-aligned small-scale QA experiment.

Retrievers receive only question text, never the QA manifest or gold labels.
Archives are read directly: no pickle loading and no extraction of arbitrary files.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import string
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import networkx as nx
import numpy as np

# This workflow is PyTorch-only, including on machines with a separate TF stack.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


VARIANTS = {"entity": {"entity"}, "entity_event": {"entity", "event"},
            "full": {"entity", "event", "concept"}}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalize_text(text):
    return " ".join(str(text).split())


def load_bundle(source: Path):
    """Read graph + provenance from a construction ZIP or directory.

    Extraction records resolve document->multiple chunk IDs. Matching uses text
    only as the bridge to hashed GraphML IDs, never a guessed substring match.
    """
    archive = ZipFile(source) if source.is_file() else None
    try:
        names = archive.namelist() if archive else [p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()]

        def read(name):
            if archive:
                if archive.getinfo(name).file_size > 1024 ** 3:
                    raise ValueError(f"Archive member exceeds 1 GiB safety limit: {name}")
                return archive.read(name)
            return (source / name).read_bytes()

        def one(filename):
            matches = [n for n in names if PurePosixPath(n).name == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {filename}, found {len(matches)}")
            return matches[0]

        graph_bytes = read(one("hotpotqa_corpus_graph.graphml"))
        graph = nx.read_graphml(io.BytesIO(graph_bytes))
        manifest = json.loads(read(one("qa_manifest.json")))
        corpus = json.loads(read(one("hotpotqa_corpus.json")))
        if not manifest or not corpus:
            raise ValueError("Empty manifest/corpus")
        if len({str(q['id']) for q in manifest}) != len(manifest):
            raise ValueError("Duplicate question IDs")
        if any(q.get("answer") is None or not (q.get("supporting_facts") or q.get("gold_document_ids")) for q in manifest):
            raise ValueError("Evaluation requires validation/train answers and supporting facts, not test labels")
        docs = {str(d["id"]): d for d in corpus}
        if len(docs) != len(corpus):
            raise ValueError("Duplicate corpus document IDs")
        text_docs = defaultdict(set)
        for doc_id, doc in docs.items():
            text_docs[normalize_text(doc["text"])].add(doc_id)
        for name in names:
            if "kg_extraction" in PurePosixPath(name).parts and name.endswith(".json"):
                raw = read(name).decode("utf-8")
                try:
                    records = json.loads(raw)
                except json.JSONDecodeError:
                    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if isinstance(records, dict):
                    records = [records]
                if not isinstance(records, list):
                    raise ValueError(f"Expected extraction list in {name}")
                for record in records:
                    doc_id = str(record.get("id", ""))
                    if doc_id in docs and record.get("original_text"):
                        text_docs[normalize_text(record["original_text"])].add(doc_id)
        passage_docs = {}
        for node, attrs in graph.nodes(data=True):
            if attrs.get("type") == "passage":
                matches = text_docs.get(normalize_text(attrs.get("id", "")), set())
                if not matches:
                    raise ValueError(f"Unmapped passage {node}; include kg_extraction JSON for chunked documents")
                passage_docs[node] = sorted(matches)
        mapped = {d for ids in passage_docs.values() for d in ids}
        if mapped != set(docs):
            raise ValueError(f"{len(set(docs) - mapped)} corpus documents have no graph passage")
        if any(set(map(str, q["document_ids"])) - set(docs) for q in manifest):
            raise ValueError("Manifest references missing corpus documents")
        if any(set(map(str, q.get("gold_document_ids", []))) - set(docs) for q in manifest):
            raise ValueError("Qrels reference missing corpus documents")
        metadata_names = [n for n in names if PurePosixPath(n).name == "dataset_metadata.json"]
        metadata = json.loads(read(metadata_names[0])) if len(metadata_names) == 1 else {}
        if metadata.get("context_mode") == "supporting":
            raise ValueError("Gold-supporting-only corpus is not a non-oracle benchmark; rebuild with --context-mode all")
        fingerprint = digest({"graph": hashlib.sha256(graph_bytes).hexdigest(), "corpus": corpus,
                              "manifest": manifest, "passage_docs": passage_docs})
        return graph, manifest, docs, passage_docs, metadata, fingerprint
    finally:
        if archive:
            archive.close()


def graph_variant(graph, variant):
    allowed = VARIANTS[variant] | {"passage"}
    # Preserve direction, parallel edges and source links, as in upstream index creation.
    return graph.subgraph([n for n, d in graph.nodes(data=True) if d.get("type") in allowed]).copy()


def normalized_embeddings(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected finite 2D embeddings")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Encoder returned a zero vector")
    return np.ascontiguousarray(array / norms)


class CachedEncoder:
    """Content-addressed, batch checkpointed cache; CPU by default on Colab."""
    def __init__(self, model, cache_dir, batch_size=32, device="cpu", revision=None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device, revision=revision)
        self.model_id = model
        self.revision = revision
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

    def encode(self, texts, query_type=None, **kwargs):
        texts = embedding_inputs(self.model_id, texts, query_type)
        arrays = []
        started = time.monotonic()
        last_log = started
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            key = digest({"model": self.model_id, "revision": self.revision, "texts": batch,
                          "normalize": True, "version": 1})
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                array = np.load(path, allow_pickle=False)
            else:
                array = normalized_embeddings(self.model.encode(batch, batch_size=self.batch_size,
                                                normalize_embeddings=True, show_progress_bar=False))
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, array, allow_pickle=False)
                temporary.replace(path)
            if len(array) != len(batch):
                raise ValueError(f"Invalid embedding cache: {path}")
            arrays.append(array)
            done = min(start + self.batch_size, len(texts))
            now = time.monotonic()
            if len(texts) > self.batch_size and (now - last_log >= 15 or done == len(texts)):
                eta = (now - started) / done * (len(texts) - done)
                print(f"  embeddings {done}/{len(texts)}, ETA ~{eta/60:.1f} min", flush=True)
                last_log = now
        return normalized_embeddings(np.concatenate(arrays))


def make_index(embeddings):
    import faiss

    embeddings = normalized_embeddings(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def index_graph(graph, encoder):
    nodes = [n for n, d in graph.nodes(data=True) if d["type"] != "passage"]
    node_set = set(nodes)
    edges = [e for e in graph.edges if e[0] in node_set and e[1] in node_set]
    if not nodes or not edges:
        raise ValueError("Graph variant has no semantic edges; cannot evaluate HippoRAG2")
    texts = {n: d["id"] for n, d in graph.nodes(data=True) if d["type"] == "passage"}
    edge_texts = [f"{graph.nodes[e[0]]['id']} {graph.edges[e]['relation']} {graph.nodes[e[1]]['id']}" for e in edges]
    print(f"Indexing {len(nodes)} nodes, {len(edges)} semantic edges, {len(texts)} passages", flush=True)
    edge_embeddings = encoder.encode(edge_texts)
    return {"KG": graph, "node_list": nodes, "edge_list": edges,
            "node_embeddings": encoder.encode([graph.nodes[n]["id"] for n in nodes]),
            "edge_embeddings": edge_embeddings, "edge_faiss_index": make_index(edge_embeddings),
            "text_embeddings": encoder.encode(list(texts.values())), "text_dict": texts}


def embedding_inputs(model, texts, query_type=None):
    if model.startswith("intfloat/multilingual-e5-"):
        prefix = "query: " if query_type is not None else "passage: "
        return [prefix + text for text in texts]
    return texts


def normalized_answer(value, language="en"):
    if language == "vi":
        value = unicodedata.normalize("NFC", str(value)).casefold()
        value = "".join(c for c in value if not unicodedata.category(c).startswith("P") and c not in string.punctuation)
        return " ".join(value.split())
    value = "".join(c for c in str(value).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def answer_scores(prediction, gold, language="en"):
    p, g = normalized_answer(prediction, language), normalized_answer(gold, language)
    booleans = {"yes", "no", "noanswer"}
    if language == "vi":
        aliases = {"yes": "có", "no": "không", "noanswer": "không có câu trả lời"}
        p, g = aliases.get(p, p), aliases.get(g, g)
        booleans = set(aliases.values())
    em = int(p == g)
    # HotpotQA official special-case: no partial credit for yes/no/noanswer.
    if p != g and ({p, g} & booleans):
        return {"em": em, "f1": 0.0}
    common = sum((Counter(p.split()) & Counter(g.split())).values())
    f1 = 2 * common / (len(p.split()) + len(g.split())) if common else 0.0
    return {"em": em, "f1": f1}


def retrieval_scores(passage_ids, sample, docs, passage_docs):
    if sample.get("gold_document_ids"):
        gold = set(map(str, sample["gold_document_ids"]))
        metrics = {}
        for k in (2, 5):
            retrieved = {d for p in passage_ids[:k] for d in passage_docs[p]}
            metrics[f"support_recall@{k}"] = len(retrieved & gold) / len(gold)
            metrics[f"all_support@{k}"] = int(gold <= retrieved)
        return metrics
    gold_titles = {str(t) for t, _ in sample["supporting_facts"]}
    metrics = {}
    for k in (2, 5):
        titles = {str(docs[d]["metadata"]["title"]) for p in passage_ids[:k] for d in passage_docs[p]}
        metrics[f"support_recall@{k}"] = len(titles & gold_titles) / len(gold_titles)
        metrics[f"all_support@{k}"] = int(gold_titles <= titles)
    return metrics


def make_hipporag2(data, encoder, reader, top_edges=30, alpha=0.9, weight=0.9, filter_edges=True,
                  candidate_selector=None):
    from atlas_rag.retriever.hipporag2 import HippoRAG2Retriever, min_max_normalize
    from atlas_rag.retriever.inference_config import InferenceConfig

    class ColabHippoRAG2(HippoRAG2Retriever):
        """Keep upstream passage seeds, PPR and ranking; tighten edge selection.

        Exact FAISS search replaces full-matrix top-k. The LLM may only select
        candidate facts: no embedding remapping of invented facts to other edges.
        Empty valid selection triggers the upstream dense-passage fallback.
        """
        def query2edge(self, query, topN=30):
            query_emb = normalized_embeddings(self.sentence_encoder.encode([query], query_type="edge"))
            _, indices = self.edge_faiss_index.search(query_emb, min(topN, len(self.edge_list)))
            candidates = [int(i) for i in indices[0] if i >= 0]
            if candidate_selector is not None:
                candidates = candidate_selector(self.edge_embeddings @ query_emb[0], topN)
            facts = [[str(self.KG.nodes[self.edge_list[i][0]]["id"]),
                      str(self.KG.edges[self.edge_list[i]]["relation"]),
                      str(self.KG.nodes[self.edge_list[i][1]]["id"])] for i in candidates]
            if self.inference_config.is_filter_edges:
                selected, filter_info = reader.filter_facts(query, facts)
            else:
                selected, filter_info = facts, {"enabled": False, "candidate_count": len(facts)}
            selected_set = {tuple(f) for f in selected}
            if selected_set - {tuple(f) for f in facts}:
                raise ValueError("Fact filter returned facts outside its candidates")
            # Same min-max normalization over the whole graph as upstream.
            scores = min_max_normalize(self.edge_embeddings @ query_emb[0])
            node_scores = defaultdict(list)
            for i, fact in zip(candidates, facts):
                if tuple(fact) in selected_set:
                    for node in self.edge_list[i][:2]:
                        node_scores[node].append(float(scores[i]))
            averaged = {n: sum(s) / len(s) for n, s in node_scores.items()}
            averaged = dict(sorted(averaged.items(), key=lambda x: x[1], reverse=True)[:self.inference_config.topk_nodes])
            self.trace = {"candidate_facts": facts, "selected_facts": selected,
                          "candidate_edges": [list(self.edge_list[i]) for i in candidates],
                          "selected_edges": [list(self.edge_list[i]) for i, f in zip(candidates, facts)
                                             if tuple(f) in selected_set],
                          "seed_nodes": averaged, "dense_fallback": not bool(averaged),
                          "filter": filter_info}
            return averaged

    config = InferenceConfig(keyword="hotpotqa", topk_edges=top_edges, topk_nodes=10,
                             ppr_alpha=alpha, weight_adjust=weight, is_filter_edges=filter_edges,
                             hipporag_mode="query2edge")
    return ColabHippoRAG2(reader, encoder, data, config)
