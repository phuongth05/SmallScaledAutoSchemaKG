"""Offline tests: real FAISS/PPR, deterministic fixture embeddings, no LLM calls."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hotpotqa_benchmark as core
import run_hotpotqa_benchmark as runner
from prepare_hotpotqa import sample_indices


class FixtureEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        # Positive finite vectors with deterministic ordering, not a quality benchmark.
        return core.normalized_embeddings([[b + 1 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts])


@pytest.fixture
def graph():
    g = nx.MultiDiGraph()
    for i in range(6):
        g.add_node(f"p{i}", id=f"Title {i}. Passage {i}", type="passage")
        g.add_node(f"n{i}", id=f"Entity {i}", type="entity", file_id=f"p{i}")
        g.add_edge(f"n{i}", f"p{i}", relation="mention in")
        if i:
            g.add_edge(f"n{i-1}", f"n{i}", relation="related to")
    g.add_node("event", id="An event happened", type="event", file_id="p0")
    g.add_node("concept", id="A general type", type="concept", file_id="p0")
    g.add_edge("n0", "event", relation="participates")
    g.add_edge("event", "p0", relation="mention in")
    g.add_edge("n0", "concept", relation="has_concept")
    g.add_edge("event", "concept", relation="has_concept")
    return g


def write_bundle(path, graph, chunked=False, supporting=False):
    corpus = [{"id": f"d{i}", "text": f"Title {i}. Passage {i}", "metadata": {"title": f"Title {i}"}} for i in range(6)]
    manifest = [{"id": "q1", "question": "Find a relation?", "answer": "yes", "document_ids": ["d0"],
                 "supporting_facts": [["Title 0", 0], ["Title 5", 0]]}]
    records = []
    if chunked:
        corpus[0]["text"] = "A longer original document"
        graph = graph.copy()
        graph.add_node("chunk2", id="A second chunk", type="passage")
        records = [{"id": "d0", "original_text": "Title 0. Passage 0"},
                   {"id": "d0", "original_text": "A second chunk"}]
    with ZipFile(path, "w") as z:
        z.writestr("kg_graphml/hotpotqa_corpus_graph.graphml", "\n".join(nx.generate_graphml(graph)))
        z.writestr("provenance/qa_manifest.json", json.dumps(manifest))
        z.writestr("provenance/hotpotqa_corpus.json", json.dumps(corpus))
        z.writestr("provenance/dataset_metadata.json", json.dumps({"context_mode": "supporting" if supporting else "all"}))
        if records:
            z.writestr("kg_extraction/results.json", "\n".join(json.dumps(r) for r in records))
        z.writestr("../../do_not_extract.txt", "ignored")
        z.writestr("untrusted.pkl", "not a pickle")
    return path


def test_bundle_chunk_provenance(tmp_path, graph):
    bundle = write_bundle(tmp_path / "bundle.zip", graph, chunked=True)
    loaded, qa, docs, mapping, metadata, fingerprint = core.load_bundle(bundle)
    assert mapping["p0"] == mapping["chunk2"] == ["d0"]
    assert len(fingerprint) == 64
    assert not (tmp_path / "do_not_extract.txt").exists()


def test_reject_oracle_corpus(tmp_path, graph):
    with pytest.raises(ValueError, match="Gold-supporting-only"):
        core.load_bundle(write_bundle(tmp_path / "bundle.zip", graph, supporting=True))


def test_variants_keep_same_passages_and_source_links(graph):
    for name, allowed in core.VARIANTS.items():
        g = core.graph_variant(graph, name)
        assert {d["type"] for _, d in g.nodes(data=True)} <= allowed | {"passage"}
        assert {n for n, d in g.nodes(data=True) if d["type"] == "passage"} == {f"p{i}" for i in range(6)}
        assert g.has_edge("n0", "p0")
    assert "concept" in graph and "event" in graph  # original untouched


def test_hotpot_scores_and_document_recall():
    assert core.answer_scores("The United States.", "United States") == {"em": 1, "f1": 1.0}
    assert core.answer_scores("yes indeed", "yes") == {"em": 0, "f1": 0.0}
    assert core.answer_scores("New York", "York") == {"em": 0, "f1": 2 / 3}
    docs = {"d1": {"metadata": {"title": "A"}}, "d2": {"metadata": {"title": "B"}}}
    mapping = {"a1": ["d1"], "a2": ["d1"], "b1": ["d2"]}
    scores = core.retrieval_scores(["a1", "a2", "b1"], {"supporting_facts": [["A", 0], ["B", 1]]}, docs, mapping)
    assert scores == {"support_recall@2": 0.5, "all_support@2": 0, "support_recall@5": 1, "all_support@5": 1}


def test_uniform_sampling():
    assert sample_indices(7405, 100, 42) == sample_indices(7405, 100, 42)
    values = sample_indices(7405, 100, 42)
    assert len(set(values)) == 100 and max(values) > 1000
    assert values != sample_indices(7405, 100, 43)
    with pytest.raises(ValueError):
        sample_indices(2, 3, 42)


def test_faiss_and_upstream_ppr_are_used(graph, monkeypatch):
    encoder = FixtureEncoder()
    data = core.index_graph(core.graph_variant(graph, "full"), encoder)
    class Filter:
        def filter_facts(self, question, facts):
            assert question == "Only question text"
            return facts[:2], {"enabled": True}
    called = []
    pagerank = nx.pagerank
    def tracked(*args, **kwargs):
        called.append(kwargs)
        return pagerank(*args, **kwargs)
    monkeypatch.setattr(nx, "pagerank", tracked)
    r = core.make_hipporag2(data, encoder, Filter())
    passages, _ = r.retrieve("Only question text", topN=5)
    assert len(passages) == 5
    assert called[0]["alpha"] == 0.9
    assert r.trace["dense_fallback"] is False
    assert len(r.trace["selected_facts"]) == 2


def test_empty_filter_explicit_dense_fallback(graph):
    encoder = FixtureEncoder()
    class Empty:
        def filter_facts(self, question, facts):
            return [], {"enabled": True}
    r = core.make_hipporag2(core.index_graph(graph, encoder), encoder, Empty())
    passages, _ = r.retrieve("Question", topN=5)
    assert len(passages) == 5 and r.trace["dense_fallback"]


def test_invented_filter_fact_rejected(graph):
    encoder = FixtureEncoder()
    class Invented:
        def filter_facts(self, question, facts):
            return [["invented", "fake", "triple"]], {}
    r = core.make_hipporag2(core.index_graph(graph, encoder), encoder, Invented())
    with pytest.raises(ValueError, match="outside"):
        r.retrieve("Question")


def args_for(source, output):
    return argparse.Namespace(source=source, output_dir=output, variants=["dense", "entity", "entity_event", "full"],
        model="fixture", base_url="http://127.0.0.1:8000/v1", model_revision=None,
        embedding_model="fixture", embedding_revision=None, embedding_device="cpu", batch_size=4,
        top_passages=5, top_edges=30, ppr_alpha=0.9, passage_weight=0.9, context_length=4096,
        max_answer_tokens=128, max_filter_tokens=1024, no_filter_edges=True, retrieval_only=True, inspect_only=False)


def test_runner_resume_and_config_guard(tmp_path, graph, monkeypatch):
    source = write_bundle(tmp_path / "bundle.zip", graph)
    args = args_for(source, tmp_path / "run")
    monkeypatch.setattr(runner, "CachedEncoder", FixtureEncoder)
    first = runner.run(args)
    assert all(m["complete"] for m in first["methods"].values())
    assert all("em" not in m["metrics"] for m in first["methods"].values())
    checkpoints = list((tmp_path / "run/results").rglob("*.json"))
    # Manifest lists only d0; all methods must nevertheless search global corpus.
    for checkpoint in checkpoints:
        record = json.loads(checkpoint.read_text())
        assert any(ids != ["d0"] for ids in record["retrieved_document_ids"])
    mtimes = [p.stat().st_mtime_ns for p in checkpoints]
    monkeypatch.setattr(runner, "CachedEncoder", lambda *a, **k: pytest.fail("resume must not load model"))
    assert runner.run(args) == first
    assert mtimes == [p.stat().st_mtime_ns for p in checkpoints]
    args.top_edges = 31
    with pytest.raises(ValueError, match="another graph/config"):
        runner.run(args)


def test_failed_answer_resumes_only_missing_question(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run")
    args.retrieval_only = False
    args.variants = ["dense", "entity"]
    calls = []
    class Reader:
        def __init__(self, args):
            pass
        def answer(self, question, passages):
            calls.append(question)
            if len(calls) == 2:
                raise ConnectionError("simulated disconnect")
            return "yes", {}
    monkeypatch.setattr(runner, "CachedEncoder", FixtureEncoder)
    monkeypatch.setattr(runner, "LocalReader", Reader)
    with pytest.raises(ConnectionError):
        runner.run(args)
    assert len(list((tmp_path / "run/results").rglob("*.json"))) == 1
    final = runner.run(args)
    assert len(calls) == 3
    assert all(m["metrics"]["em"] == 1 for m in final["methods"].values())


def test_new_notebook_code_syntax():
    path = ROOT / "colab/AutoSchemaKG_HotpotQA_PaperAligned.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_reader_rejects_truncated_generation():
    from types import SimpleNamespace as NS
    reader = runner.LocalReader.__new__(runner.LocalReader)
    reader.args = NS(context_length=100, model="fixture")
    reader.token_count = lambda messages: 10
    reader.client = NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[NS(
        finish_reason="length", message=NS(content="partial answer"))]))))
    with pytest.raises(RuntimeError, match="Incomplete LLM"):
        reader.generate([], 10)


def test_fact_filter_context_guard_and_candidate_validation():
    from types import SimpleNamespace as NS
    reader = runner.LocalReader.__new__(runner.LocalReader)
    reader.args = NS(context_length=160, max_filter_tokens=10)
    reader.token_count = lambda messages: len(messages[-1]["content"])
    facts = [["a", "b", "c"], ["d" * 100, "e", "f"]]
    reader.generate = lambda messages, budget: ('{"selected_ids": [0]}', 70)
    selected, info = reader.filter_facts("q", facts)
    assert selected == facts[:1] and info["dropped_for_context"] == 1
    reader.generate = lambda messages, budget: ('{"selected_ids": [99]}', 70)
    with pytest.raises(ValueError, match="retries exhausted"):
        reader.filter_facts("q", facts)


def test_random_api_sampling_fetches_each_page_once(monkeypatch):
    from types import SimpleNamespace as NS
    import requests
    from prepare_hotpotqa import load_random_examples
    calls = []
    class Session:
        def mount(self, *args):
            pass
        def close(self):
            pass
        def get(self, url, params, timeout):
            offset, length = params["offset"], params["length"]
            calls.append((offset, length))
            return NS(raise_for_status=lambda: None, json=lambda: {"num_rows_total": 250,
                "rows": [{"row_idx": i, "row": {"id": str(i)}} for i in range(offset, offset + length)]})
    monkeypatch.setattr(requests, "Session", Session)
    args = NS(loader="api", config="distractor", split="validation", max_questions=20, seed=42)
    result = list(load_random_examples(args))
    assert [int(r["id"]) for r in result] == sample_indices(250, 20, 42)
    assert len(calls[1:]) == len(set(offset for offset, _ in calls[1:]))
