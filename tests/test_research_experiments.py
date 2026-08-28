"""Offline research protocol tests: real PPR/FAISS, fixture encoder, no network/GPU."""
import ast
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_experiments as research
import run_research_experiments as runner
from report_research_experiments import report
from test_hotpotqa_v2 import FixtureEncoder, graph, write_bundle


def args_for(source, output, *extra):
    return runner.parse_args([str(source), "--output-dir", str(output), "--embedding-model", "fixture", *extra])


def test_stratified_split_disjoint_reproducible_order_independent():
    qa = [{"id": str(i), "type": "bridge" if i < 80 else "comparison"} for i in range(100)]
    split = research.split_questions(qa, 20, 42)
    assert split == research.split_questions(list(reversed(qa)), 20, 42)
    assert len(split["dev"]) == 20 and len(split["test"]) == 80
    assert not set(split["dev"]) & set(split["test"])
    assert sum(int(i) < 80 for i in split["dev"]) == 16
    assert set(split["smoke"]) <= set(split["dev"])


@pytest.mark.parametrize("mutation", [{"name": "../bad"}, {"concept_cap": -1}, {"concept_weight": 0},
                                      {"candidate_policy": "typo"}, {"pruning": "typo"}, {"unknown": 1}])
def test_suite_rejects_bad_settings(mutation):
    with pytest.raises(ValueError):
        research.validate_arms([{"name": "full", "method": "graph", **mutation}])


def test_pruning_preserves_core_and_matched_random_edge_budget(graph):
    for i in range(4):
        graph.add_node(f"c{i}", id=f"concept{i}", type="concept", file_id="p0")
        graph.add_edge("n0", f"c{i}", relation="has_concept")
        graph.add_edge(f"c{i}", "n0", relation="has_concept")
    before = graph.copy()
    cap, stats = research.prepare_graph(graph, {"variant": "full", "concept_cap": 1})
    random_g, random_stats = research.prepare_graph(graph, {"variant": "full", "concept_cap": 1, "pruning": "random_matched"})
    assert stats["concept_edges_removed"] == random_stats["concept_edges_removed"] > 0
    assert cap.number_of_edges() == random_g.number_of_edges()
    expected = {n for n, d in graph.nodes(data=True) if d["type"] != "concept"}
    assert expected <= set(cap) and expected <= set(random_g)
    assert nx.utils.graphs_equal(graph, before)
    assert nx.utils.graphs_equal(random_g, research.prepare_graph(graph, {"variant": "full", "concept_cap": 1, "pruning": "random_matched"})[0])
    core_edges = {e for e in graph.edges if not research.is_concept_edge(graph, e)}
    assert core_edges <= set(cap.edges) and core_edges <= set(random_g.edges)


def test_weight_only_concept_edges_and_actual_ppr(graph, monkeypatch):
    g, _ = research.prepare_graph(graph, {"concept_weight": 0.25})
    assert g["n0"]["concept"][0]["weight"] == 0.25
    assert "weight" not in g["n0"]["p0"][0]
    called = []
    original = nx.pagerank
    def tracking(g, **kwargs):
        called.append(g["n0"]["concept"][0]["weight"])
        return original(g, **kwargs)
    monkeypatch.setattr(nx, "pagerank", tracking)
    encoder = FixtureEncoder()
    data = runner.index_graph(g, encoder)
    r = runner.make_hipporag2(data, encoder, None, filter_edges=False)
    assert len(r.retrieve("question")[0]) == 5
    assert called == [0.25]


def test_candidate_quota_backfill_and_factual_only(graph):
    edges = [e for e in graph.edges if all(graph.nodes[n]["type"] != "passage" for n in e[:2])]
    data = {"KG": graph, "edge_list": edges}
    scores = np.array([10 if research.is_concept_edge(graph, e) else i for i, e in enumerate(edges)])
    chosen = research.candidate_selector(data, "quota", 2)(scores, 3)
    assert len(chosen) == len(set(chosen)) == 3
    assert sum(not research.is_concept_edge(graph, edges[i]) for i in chosen) == 2
    chosen = research.candidate_selector(data, "quota", 0)(scores, 5)
    assert len(chosen) == 5  # Only two concept edges: backfill factual.
    chosen = research.candidate_selector(data, "factual")(scores, 30)
    assert all(not research.is_concept_edge(graph, edges[i]) for i in chosen)
    assert research.candidate_selector(data, "all") is None


def test_bm25_unicode_and_empty_query_stable():
    bm25 = research.BM25({"a": "Hà Nội là thủ đô", "b": "Paris ở Pháp", "c": ""})
    assert bm25.retrieve("Hà Nội", 2)[0] == "a"
    assert bm25.retrieve("Paris", 2)[0] == "b"
    assert bm25.retrieve("", 2) == ["a", "b"]


def test_k10_scores_and_oracle_dedup_chunks():
    docs = {f"d{i}": {"metadata": {"title": str(i)}} for i in range(10)}
    mapping = {f"p{i}": [f"d{i}"] for i in range(10)}
    q = {"gold_document_ids": ["d0", "d9"]}
    scores = research.metrics_at_k(list(mapping), q, docs, mapping)
    assert scores["support_recall@5"] == .5 and scores["all_support@10"] == 1
    assert research.oracle_passages(q, docs, mapping, 5) == ["p0", "p9"]
    with pytest.raises(ValueError, match="budget"):
        research.oracle_passages(q, docs, mapping, 1)


def test_audit_has_source_and_strata(graph):
    audit = research.audit_template(graph, {f"p{i}": [f"d{i}"] for i in range(6)}, 2)
    assert audit == research.audit_template(graph, {f"p{i}": [f"d{i}"] for i in range(6)}, 2)
    assert "concept" in audit["stratum_population"]
    assert all(row["candidate_evidence"] and row["reviewer_1"]["supported"] is None for row in audit["rows"])


def test_plan_never_loads_models_and_preserves_audit(tmp_path, graph, monkeypatch):
    source = write_bundle(tmp_path / "bundle.zip", graph)
    args = args_for(source, tmp_path / "run", "--audit-per-type", "2")
    monkeypatch.setattr(runner, "CachedEncoder", lambda *a, **k: pytest.fail("No download in plan"))
    monkeypatch.setattr(runner, "LocalReader", lambda *a, **k: pytest.fail("No server in plan"))
    plan = runner.run(args)
    assert plan["questions_per_arm"] == 1 and not list((tmp_path / "run").rglob("results"))
    path = tmp_path / "run/audit_template.json"
    path.write_text('{"human_annotation": "preserve"}', encoding="utf-8")
    runner.run(args)
    assert json.loads(path.read_text()) == {"human_annotation": "preserve"}


def test_all_diagnostic_arms_resume_and_report(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run", "--execute")
    monkeypatch.setattr(runner, "CachedEncoder", FixtureEncoder)
    monkeypatch.setattr(runner, "LocalReader", lambda *a, **k: pytest.fail("Diagnostic must not use LLM"))
    result = runner.run(args)
    assert len(result) == 11 and all(s["complete"] for s in result.values())
    paths = list((tmp_path / "run/diagnostic").rglob("results/*.json"))
    mtimes = [p.stat().st_mtime_ns for p in paths]
    assert all("em" not in s["metrics"] for s in result.values())
    monkeypatch.setattr(runner, "CachedEncoder", lambda *a, **k: pytest.fail("Completed resume must not load models"))
    runner.run(args)
    assert mtimes == [p.stat().st_mtime_ns for p in paths]
    summary = report(tmp_path / "run", repetitions=100)
    assert len(summary["results"]) == 11 and summary["baseline_available"]
    assert "full" in summary["paired_vs_baseline"]
    args.top_edges += 1
    with pytest.raises(ValueError, match="Immutable"):
        runner.run(args)


def test_qa_oracle_no_context_and_failure_resume(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run",
                    "--execute", "--stage", "qa", "--arms", "bm25", "oracle", "no_context")
    calls = []
    class Reader:
        def __init__(self, args):
            pass
        def answer(self, question, passages):
            calls.append(list(passages))
            if len(calls) == 2:
                raise ConnectionError("simulated disconnect")
            return "yes", {"truncated_passages": 0, "usage": {"total_tokens": 10}}
    monkeypatch.setattr(runner, "LocalReader", Reader)
    monkeypatch.setattr(runner, "CachedEncoder", lambda *a, **k: pytest.fail("These arms need no embeddings"))
    with pytest.raises(ConnectionError):
        runner.run(args)
    assert not (tmp_path / "run/.research.lock").exists()
    result = runner.run(args)
    assert len(calls) == 4 and calls[-1] == []
    assert result["oracle"]["metrics"]["em"] == 1
    summary = report(tmp_path / "run", stage="qa", baseline="bm25", repetitions=100)
    assert set(summary["results"]) == {"bm25"}
    assert set(summary["reader_diagnostics_not_competitive"]) == {"oracle", "no_context"}


def test_qa_filter_ablation_receives_question_only(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run",
                    "--execute", "--stage", "qa", "--arms", "full", "full_no_filter")
    filters = []
    class Reader:
        def __init__(self, args):
            pass
        def filter_facts(self, question, facts):
            assert question == "Find a relation?"  # No gold answer/document IDs passed.
            filters.append(facts)
            return facts[:2], {"enabled": True, "dropped_for_context": 0}
        def answer(self, question, passages):
            assert question == "Find a relation?" and len(passages) == 5
            return "yes", {"context_passages": passages}
    monkeypatch.setattr(runner, "LocalReader", Reader)
    monkeypatch.setattr(runner, "CachedEncoder", FixtureEncoder)
    results = runner.run(args)
    assert len(filters) == 1
    assert all(s["metrics"]["em"] == 1 for s in results.values())


def test_report_excludes_partial_and_rejects_corruption(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run",
                    "--execute", "--arms", "bm25")
    runner.run(args)
    checkpoint = next((tmp_path / "run/diagnostic").rglob("results/*.json"))
    original = checkpoint.read_text(encoding="utf-8")
    checkpoint.unlink()
    result = report(tmp_path / "run", baseline="bm25", repetitions=100)
    assert result["results"] == {} and result["incomplete_excluded"]["bm25"]["completed"] == 0
    row = json.loads(original)
    row["config_id"] = "wrong"
    checkpoint.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        report(tmp_path / "run", baseline="bm25", repetitions=100)


def test_test_split_requires_explicit_authorization(tmp_path):
    args = args_for(tmp_path / "missing.zip", tmp_path / "run", "--execute", "--split", "test")
    with pytest.raises(ValueError, match="allow-test"):
        runner.run(args)


def test_lock_blocks_plan_writes_without_removing_existing_lock(tmp_path, graph):
    output = tmp_path / "run"
    output.mkdir()
    lock = output / ".research.lock"
    lock.write_text("another process", encoding="utf-8")
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), output)
    with pytest.raises(RuntimeError, match="locked"):
        runner.run(args)
    assert lock.read_text() == "another process"
    assert not (output / "protocol.json").exists()


def test_paired_bootstrap_alignment():
    with pytest.raises(ValueError):
        research.paired_bootstrap({"a": 1}, {"b": 1})
    result = research.paired_bootstrap({"a": 0, "b": 0}, {"a": 1, "b": 1}, repetitions=100)
    assert result["delta"] == 1 and result["ci95"] == [1, 1]


def test_prepared_corpus_bm25_without_kg(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    for name, value in {
        "hotpotqa_corpus.json": [{"id": "d", "text": "Hà Nội", "metadata": {"title": "Hà Nội"}}],
        "qa_manifest.json": [{"id": "q", "question": "Hà Nội", "answer": "có", "gold_document_ids": ["d"]}],
        "dataset_metadata.json": {"language": "vi", "context_mode": "all"},
    }.items():
        (source / name).write_text(json.dumps(value), encoding="utf-8")
    args = args_for(source, tmp_path / "run", "--arms", "bm25", "--execute")
    monkeypatch.setattr(runner, "CachedEncoder", lambda *a, **k: pytest.fail("BM25 needs no embeddings"))
    result = runner.run(args)
    assert result["bm25"]["metrics"]["all_support@5"] == 1
    args.arms = ["full"]
    with pytest.raises(ValueError, match="No constructed KG"):
        runner.run(args)


def test_new_notebook_syntax():
    notebook = ROOT / "colab/AutoSchemaKG_Research.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
