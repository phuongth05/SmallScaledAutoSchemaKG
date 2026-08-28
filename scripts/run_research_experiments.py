"""Plan (default) or execute reproducible HotpotQA research arms from saved data.

Never builds a KG or starts a GPU server. Use --execute explicitly for inference.
Prepared corpus directories also support BM25/dense/oracle without a built KG.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import networkx as nx

from hotpotqa_benchmark import (CachedEncoder, answer_scores, atomic_json, digest,
    index_graph, load_bundle, make_hipporag2, make_index)
from run_hotpotqa_benchmark import LocalReader
from research_experiments import (BM25, audit_template, candidate_selector, edge_passages,
    metrics_at_k, oracle_passages, prepare_graph, split_questions, support_documents, validate_arms)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Construction ZIP/directory, or prepared corpus directory for non-graph arms")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/research"))
    p.add_argument("--suite", type=Path, default=ROOT / "research/experiments.json")
    p.add_argument("--arms", nargs="+", help="Default all arms available in the stage")
    p.add_argument("--stage", choices=("diagnostic", "qa"), default="diagnostic")
    p.add_argument("--split", choices=("smoke", "dev", "test"), default="smoke")
    p.add_argument("--dev-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-test", action="store_true", help="Explicitly authorize final test evaluation after freezing dev choices")
    p.add_argument("--execute", action="store_true", help="Without this flag: validate and save plan only, no models/server")
    p.add_argument("--audit-per-type", type=int, default=0, help="Export a human-review template without overwriting existing annotations")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--model-revision")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--embedding-model", help="Default MiniLM for EN, multilingual-e5-small for VI")
    p.add_argument("--embedding-revision")
    p.add_argument("--embedding-device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-edges", type=int, default=30)
    p.add_argument("--retrieve-k", type=int, default=10)
    p.add_argument("--answer-k", type=int, default=5)
    p.add_argument("--ppr-alpha", type=float, default=0.9)
    p.add_argument("--passage-weight", type=float, default=0.9)
    p.add_argument("--context-length", type=int, default=4096)
    p.add_argument("--max-answer-tokens", type=int, default=128)
    p.add_argument("--max-filter-tokens", type=int, default=1024)
    p.add_argument("--filter-max-attempts", type=int, default=2)
    p.add_argument("--filter-failure-policy", choices=("error", "dense"), default="error")
    return p.parse_args(argv)


def load_source(source):
    if source.is_file() or any(source.rglob("hotpotqa_corpus_graph.graphml")):
        return (*load_bundle(source), True)
    def read(name):
        matches = list(source.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name} in prepared directory")
        return json.loads(matches[0].read_text(encoding="utf-8"))
    corpus, manifest, metadata = read("hotpotqa_corpus.json"), read("qa_manifest.json"), read("dataset_metadata.json")
    docs = {str(d["id"]): d for d in corpus}
    if not corpus or not manifest or len(docs) != len(corpus):
        raise ValueError("Empty data or duplicate documents")
    if len({str(q["id"]) for q in manifest}) != len(manifest):
        raise ValueError("Duplicate question IDs")
    if metadata.get("context_mode") == "supporting":
        raise ValueError("Gold-supporting-only corpus is not a non-oracle benchmark")
    graph, mapping = nx.DiGraph(), {}
    for d, row in docs.items():
        node = digest(["document", d])
        graph.add_node(node, id=row["text"], type="passage")
        mapping[node] = [d]
    for q in manifest:
        if q.get("answer") is None or not (q.get("gold_document_ids") or q.get("supporting_facts")):
            raise ValueError("Missing evaluation labels")
        gold = support_documents(q, docs)
        if not gold or gold - set(docs):
            raise ValueError("Unresolved supporting documents")
    return graph, manifest, docs, mapping, metadata, digest([corpus, manifest, metadata]), False


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Immutable configuration mismatch: {path}; use a NEW output directory")
    else:
        atomic_json(path, value)


def summarize(directory, ids, config_id):
    records = []
    for qid in ids:
        path = directory / "results" / f"{digest(qid)}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["config_id"] != config_id or str(row["id"]) != qid:
                raise ValueError("Checkpoint mismatch")
            records.append(row)
    keys = records[0]["metrics"] if records else []
    return {"config_id": config_id, "expected": len(ids), "completed": len(records),
            "complete": len(records) == len(ids),
            "metrics": {k: sum(r["metrics"][k] for r in records) / len(records) for k in keys}}


def run(args):
    if min(args.top_edges, args.batch_size, args.answer_k, args.max_answer_tokens, args.max_filter_tokens) <= 0:
        raise ValueError("Counts must be positive")
    if args.retrieve_k < max(10, args.answer_k) or not 0 < args.ppr_alpha < 1 or args.passage_weight <= 0:
        raise ValueError("Require retrieve-k >= max(10, answer-k), 0 < alpha < 1, positive passage weight")
    if args.context_length <= max(args.max_answer_tokens, args.max_filter_tokens) + 32:
        raise ValueError("Context budget too small")
    if args.execute and args.split == "test" and not args.allow_test:
        raise ValueError("Freeze dev choices first, then explicitly pass --allow-test")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported suite schema")
    all_arms = validate_arms(suite["arms"])
    available = [a for a in all_arms if args.stage in a.get("stages", ["diagnostic", "qa"])]
    if args.arms and (len(set(args.arms)) != len(args.arms) or set(args.arms) - {a["name"] for a in available}):
        raise ValueError("Unknown, duplicate or stage-incompatible arms")
    arms = [a for a in available if args.arms is None or a["name"] in args.arms]
    if not arms:
        raise ValueError("No arms selected")
    if any(a.get("candidate_policy") == "quota" and a.get("factual_quota", 20) > args.top_edges for a in arms):
        raise ValueError("factual_quota cannot exceed top_edges")
    graph, manifest, docs, mapping, metadata, fingerprint, has_kg = load_source(args.source)
    corpus_titles = {str(d["metadata"]["title"]) for d in docs.values()}
    for sample in manifest:
        if not sample.get("gold_document_ids"):
            if {str(t) for t, _ in sample["supporting_facts"]} - corpus_titles:
                raise ValueError("Supporting title missing from corpus; cannot compare against full evidence")
    if not has_kg and any(a["method"] == "graph" for a in arms):
        raise ValueError("No constructed KG: select --arms bm25 dense (or QA oracle/no_context)")
    args.language = metadata.get("language", "en")
    if args.language not in {"en", "vi"}:
        raise ValueError("Unsupported scoring language")
    if args.embedding_model is None:
        args.embedding_model = ("intfloat/multilingual-e5-small" if args.language == "vi"
                                else "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    if args.language == "vi" and args.embedding_model.startswith("sentence-transformers/multi-qa-MiniLM"):
        raise ValueError("Use a multilingual encoder for VI")
    split = split_questions(manifest, args.dev_size, args.seed)
    ids = split[args.split]
    if not ids:
        raise ValueError("Empty split; provide more questions or reduce dev-size")
    by_id = {str(q["id"]): q for q in manifest}
    dev_docs = set().union(*(support_documents(by_id[q], docs) for q in split["dev"]))
    test_docs = set().union(*(support_documents(by_id[q], docs) for q in split["test"]))
    split["shared_support_documents_dev_test"] = len(dev_docs & test_docs)
    excluded = {"source", "output_dir", "suite", "arms", "stage", "split", "execute", "allow_test", "base_url", "audit_per_type"}
    settings = {k: v for k, v in vars(args).items() if k not in excluded}
    files = [Path(__file__), ROOT / "scripts/research_experiments.py", ROOT / "scripts/hotpotqa_benchmark.py",
             ROOT / "scripts/run_hotpotqa_benchmark.py", ROOT / "atlas_rag/retriever/hipporag2.py",
             ROOT / "atlas_rag/retriever/inference_config.py", ROOT / "atlas_rag/llm_generator/prompt/rag_prompt.py",
             ROOT / "scripts/colab_v2_utils.py"]
    versions = {}
    for package in ("numpy", "networkx", "scipy", "faiss-cpu", "sentence-transformers", "torch", "transformers", "openai", "json-repair"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    protocol = {"version": 1, "dataset_fingerprint": fingerprint, "dataset_metadata": metadata,
                "has_constructed_kg": has_kg, "settings": settings, "suite": suite, "splits": split,
                "code": {p.name: digest(p.read_text(encoding="utf-8")) for p in files}, "versions": versions}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".research.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Output locked. Stop other run; remove .research.lock only after confirming no process is running.") from exc
    os.close(fd)
    try:
        # Lock before writing protocol/plan too: simultaneous startup must not race
        # atomic_json's temporary files or freeze two different protocols.
        freeze_json(output / "protocol.json", protocol)
        freeze_json(output / "splits.json", split)
        plan = {"stage": args.stage, "split": args.split, "questions_per_arm": len(ids), "arms": arms,
                "documents": len(docs), "passages": len(mapping), "executes_llm": args.stage == "qa",
                "note": "All diagnostic graph arms disable LLM filtering. Oracle is an upper-bound diagnostic only. No construction calls."}
        atomic_json(output / f"plan_{args.stage}_{args.split}.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.audit_per_type > 0:
            if not has_kg:
                raise ValueError("Audit requires a constructed KG")
            audit_path = output / "audit_template.json"
            if not audit_path.exists():
                atomic_json(audit_path, audit_template(graph, mapping, args.audit_per_type, args.seed))
            else:
                print("Existing audit template preserved (may contain human annotations).", flush=True)
        if not args.execute:
            print("PLAN ONLY. Add --execute to run. No model loaded or server contacted.", flush=True)
            return plan
        return execute(args, output, protocol, arms, ids, by_id, graph, docs, mapping)
    finally:
        lock.unlink()


def execute(args, output, protocol, arms, ids, by_id, graph, docs, mapping):
    texts = {p: graph.nodes[p]["id"] for p in mapping}
    text_ids = list(texts)
    reader, encoder, dense_index, lexical = None, None, None, None
    summaries = {}
    for arm in arms:
        folder = output / args.stage / args.split / arm["name"]
        config = {"protocol_id": digest(protocol), "arm": arm, "stage": args.stage, "split": args.split,
                  "question_ids": ids, "uses_gold_for_retrieval": arm["method"] == "oracle"}
        freeze_json(folder / "config.json", config)
        config_id = digest(config)
        summary = summarize(folder, ids, config_id)
        atomic_json(folder / "summary.json", summary)
        pending = [qid for qid in ids if not (folder / "results" / f"{digest(qid)}.json").exists()]
        if not pending:
            summaries[arm["name"]] = summary
            print(f"{arm['name']}: complete; skipped", flush=True)
            continue
        setup_started = time.monotonic()
        method, retriever, stats = arm["method"], None, None
        if args.stage == "qa" and reader is None:
            reader = LocalReader(args)
        if method in {"dense", "graph"}:
            if encoder is None:
                encoder = CachedEncoder(args.embedding_model, output / "embedding_cache", args.batch_size,
                                        args.embedding_device, args.embedding_revision)
            if dense_index is None:
                dense_index = make_index(encoder.encode(list(texts.values())))
        if method == "bm25" and lexical is None:
            lexical = BM25(texts)
        if method == "graph":
            modified, stats = prepare_graph(graph, arm, args.seed)
            data = index_graph(modified, encoder)
            selector = candidate_selector(data, arm.get("candidate_policy", "all"), arm.get("factual_quota", 20))
            retriever = make_hipporag2(data, encoder, reader, args.top_edges, args.ppr_alpha,
                args.passage_weight, args.stage == "qa" and arm.get("filter", True), selector)
            text_to_id = {v: k for k, v in texts.items()}
            if len(text_to_id) != len(texts):
                raise ValueError("Graph retrieval requires unique passage texts")
        session = {"time_unix": time.time(), "setup_seconds": time.monotonic() - setup_started,
                   "pending_questions": len(pending), "graph": stats,
                   "cache_policy": "Shared content-addressed embedding cache; setup time is NOT a cold-build comparison",
                   "python": sys.version}
        session_path = folder / "sessions.json"
        sessions = json.loads(session_path.read_text(encoding="utf-8")) if session_path.exists() else []
        sessions.append(session)
        atomic_json(session_path, sessions)
        loop_start = time.monotonic()
        for i, qid in enumerate(pending, 1):
            sample = by_id[qid]
            started = time.monotonic()
            trace = {"method": method}
            print(f"[{args.stage}/{args.split}/{arm['name']} {i}/{len(pending)}] {qid}", flush=True)
            try:
                # Normal retrieval receives QUESTION TEXT ONLY. Oracle is explicitly separate.
                question = str(sample["question"])
                if method == "bm25":
                    retrieved = lexical.retrieve(question, args.retrieve_k)
                elif method == "dense":
                    _, indices = dense_index.search(encoder.encode([question], query_type="passage"), min(args.retrieve_k, len(text_ids)))
                    retrieved = [text_ids[int(j)] for j in indices[0] if j >= 0]
                elif method == "oracle":
                    retrieved = oracle_passages(sample, docs, mapping, args.answer_k)
                    trace["uses_gold_labels"] = True
                elif method == "no_context":
                    retrieved = []
                else:
                    passages, _ = retriever.retrieve(question, topN=args.retrieve_k)
                    retrieved = [text_to_id[p] for p in passages]
                    trace = retriever.trace
                retrieval_seconds = time.monotonic() - started
                if not retrieved and method != "no_context":
                    raise RuntimeError("No retrieved evidence")
                context_ids = retrieved[:args.answer_k]
                passages = [texts[p] for p in context_ids]
                before_answer = time.monotonic()
                prediction, info = (None, None) if args.stage == "diagnostic" else reader.answer(question, passages)
                answer_seconds = time.monotonic() - before_answer
                # Labels first used here for all non-oracle arms.
                metrics = {} if method == "no_context" else metrics_at_k(retrieved, sample, docs, mapping)
                if prediction is not None:
                    metrics.update(answer_scores(prediction, sample["answer"], args.language))
                coverage = {}
                if method == "graph":
                    for phase in ("candidate", "selected"):
                        evidence = edge_passages(graph, trace.get(f"{phase}_edges", []), mapping)
                        coverage[phase] = metrics_at_k(evidence, sample, docs, mapping, ks=(len(evidence),))
                    trace["direct_provenance_coverage"] = coverage
                row = {"config_id": config_id, "id": qid, "question": question,
                       "question_type": sample.get("type", "unknown"), "method": arm["name"],
                       "gold_answer": sample["answer"], "prediction": prediction, "metrics": metrics,
                       "retrieved_passage_ids": retrieved, "retrieved_document_ids": [mapping[p] for p in retrieved],
                       "retrieved_titles": [[docs[d]["metadata"]["title"] for d in mapping[p]] for p in retrieved],
                       "context_passage_ids": context_ids, "reader": info, "retrieval_trace": trace,
                       "context_document_support": (metrics_at_k(context_ids, sample, docs, mapping,
                            ks=(args.answer_k,)) if context_ids else {}),
                       "timing": {"retrieval_seconds": retrieval_seconds, "answer_seconds": answer_seconds,
                                  "total_seconds": time.monotonic() - started}}
                atomic_json(folder / "results" / f"{digest(qid)}.json", row)
                n = summary["completed"]
                summary["metrics"] = {k: (summary["metrics"].get(k, 0) * n + v) / (n + 1) for k, v in metrics.items()}
                summary["completed"] = n + 1
                summary["complete"] = n + 1 == len(ids)
                atomic_json(folder / "summary.json", summary)
            except Exception as exc:
                atomic_json(folder / "last_error.json", {"question_id": qid, "error": str(exc),
                    "filter_diagnostics": getattr(exc, "diagnostics", None),
                    "resume": "Rerun identical arguments. Completed question checkpoints are preserved."})
                raise
            eta = (time.monotonic() - loop_start) / i * (len(pending) - i)
            print(f"  {metrics}; arm ETA ~{eta/60:.1f} min (excludes later arms/setup)", flush=True)
        summaries[arm["name"]] = summary
    return summaries


if __name__ == "__main__":
    run(parse_args())
