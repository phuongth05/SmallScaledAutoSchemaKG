"""Vietnamese adapter tests; no GPU, model downloads or source dataset copies."""
import argparse
import ast
import importlib.util
import json
import sys
import types
import unicodedata
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_hotpotqa_vn import load_final, prepare
from hotpotqa_benchmark import answer_scores, embedding_inputs, load_bundle, retrieval_scores
from run_hotpotqa_vn import construction_args
from audit_vn_extraction_pilot import load_records, make_audit, make_concentration, parse_log
from run_colab_v1 import inspect_extraction_progress, write_extraction_progress

validation_spec = importlib.util.spec_from_file_location(
    "extraction_validation",
    ROOT / "atlas_rag/kg_construction/extraction_validation.py",
)
validation_module = importlib.util.module_from_spec(validation_spec)
validation_spec.loader.exec_module(validation_module)
sanitize_vietnamese_event_relations = validation_module.sanitize_vietnamese_event_relations

json_to_csv_spec = importlib.util.spec_from_file_location(
    "json_to_csv", ROOT / "atlas_rag/kg_construction/utils/json_processing/json_to_csv.py",
)
json_to_csv_module = importlib.util.module_from_spec(json_to_csv_spec)
json_to_csv_spec.loader.exec_module(json_to_csv_module)
json2csv = json_to_csv_module.json2csv


@pytest.fixture
def final_dir(tmp_path):
    root = tmp_path / "final"
    root.mkdir()
    queries = [{"id": f"q{i}", "question_vi": f"Câu hỏi {i}?", "answer_vi": "có",
                "question_en": "DO NOT INJECT QUESTION", "answer_en": "DO NOT INJECT ANSWER",
                "type": "comparison", "level": "hard",
                "supporting_facts": [{"title": "Different original English title", "sent_id": 0}]}
               for i in range(2)]
    docs = [{"id": str(i), "title": f"Tiêu đề {i}", "text": f"Đây là đoạn văn số {i}.", "language": "vi"}
            for i in range(5)]
    for name, rows in [("queries.jsonl", queries), ("corpus.jsonl", docs)]:
        (root / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (root / "qrels.tsv").write_text("query_id\tcorpus_id\tscore\nq0\t0\t1\nq0\t1\t1\nq1\t1\t1\nq1\t2\t1\n", encoding="utf-8")
    (root / "source_qa_issues.csv").write_text("id,notes\nq1,keep this query\n", encoding="utf-8")
    return root


def test_prepare_keeps_global_corpus_and_vi_labels(final_dir, tmp_path):
    output = tmp_path / "prepared"
    metadata = prepare(final_dir, output, max_questions=1, sampling="sequential")
    docs = json.loads((output / "hotpotqa_corpus.json").read_text(encoding="utf-8"))
    qa = json.loads((output / "qa_manifest.json").read_text(encoding="utf-8"))
    assert len(docs) == 5 and len(qa) == 1
    assert qa[0]["answer"] == "có" and qa[0]["question"] == "Câu hỏi 0?"
    assert qa[0]["document_ids"] == []
    assert qa[0]["gold_document_ids"] == ["0", "1"]
    assert all(d["metadata"]["lang"] == "vi" for d in docs)
    assert "DO NOT INJECT" not in json.dumps(docs)
    assert "gold_document_ids" not in json.dumps(docs)
    assert metadata["source_qa_issue_ids"] == ["q1"]
    assert prepare(final_dir, output, max_questions=1, sampling="sequential") == metadata
    with pytest.raises(FileExistsError):
        prepare(final_dir, output, max_questions=2)


def test_final_qrels_integrity(final_dir):
    qrels = final_dir / "qrels.tsv"
    qrels.write_text(qrels.read_text() + "q0\t999\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dangling qrel"):
        load_final(final_dir)


def test_missing_vi_does_not_fall_back_to_english(final_dir):
    path = final_dir / "queries.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["answer_vi"] = ""
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing English fallback"):
        load_final(final_dir)


def test_qrels_ids_override_original_support_titles():
    sample = {"gold_document_ids": ["0", "1"], "supporting_facts": [["not a Vietnamese corpus title", 0]]}
    mapping = {"a": ["0"], "b": ["0"], "c": ["1"]}
    scores = retrieval_scores(["a", "b", "c"], sample, {}, mapping)
    assert scores["support_recall@2"] == 0.5
    assert scores["all_support@2"] == 0
    assert scores["all_support@5"] == 1


def test_vietnamese_scoring_preserves_accents_and_articles():
    assert answer_scores("CÓ!", "có", "vi") == {"em": 1, "f1": 1}
    assert answer_scores("yes", "có", "vi")["em"] == 1
    assert answer_scores("không có", "không", "vi")["f1"] == 0
    assert answer_scores(unicodedata.normalize("NFD", "Hà Nội"), "Hà Nội", "vi")["em"] == 1
    assert answer_scores("Ha Noi", "Hà Nội", "vi")["em"] == 0
    assert answer_scores("An", "", "vi")["em"] == 0  # no English article removal
    assert answer_scores("the United States", "United States")["em"] == 1  # EN unchanged


def test_e5_query_and_document_prefixes():
    model = "intfloat/multilingual-e5-small"
    assert embedding_inputs(model, ["Hà Nội"]) == ["passage: Hà Nội"]
    for kind in ("edge", "passage", "entity"):
        assert embedding_inputs(model, ["Ở đâu?"], kind) == ["query: Ở đâu?"]
    assert embedding_inputs("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", ["Hi"], "edge") == ["Hi"]


def test_vi_prompt_registry_and_json_contract():
    from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS, CONCEPT_INSTRUCTIONS
    assert set(TRIPLE_INSTRUCTIONS["vi"]) == set(TRIPLE_INSTRUCTIONS["en"])
    assert set(CONCEPT_INSTRUCTIONS["vi"]) == set(CONCEPT_INSTRUCTIONS["en"])
    assert '"Head"' in TRIPLE_INSTRUCTIONS["vi"]["entity_relation"]
    assert '"Entity"' in TRIPLE_INSTRUCTIONS["vi"]["event_entity"]
    for kind in ("entity", "event", "relation"):
        assert f"[{kind.upper()}]" in CONCEPT_INSTRUCTIONS["vi"][kind]
    assert "[CONTEXT]" in CONCEPT_INSTRUCTIONS["vi"]["entity"]


def test_vietnamese_bundle_compatible_with_v2(final_dir, tmp_path):
    output = tmp_path / "bundle"
    prepare(final_dir, output, max_questions=1)
    docs = json.loads((output / "hotpotqa_corpus.json").read_text(encoding="utf-8"))
    g = nx.DiGraph()
    for d in docs:
        g.add_node("p" + d["id"], type="passage", id=d["text"])
    nx.write_graphml(g, output / "hotpotqa_corpus_graph.graphml")
    graph, qa, corpus, mapping, metadata, fingerprint = load_bundle(output)
    assert metadata["language"] == "vi"
    assert len(mapping) == 5
    assert qa[0]["gold_document_ids"]
    assert len(fingerprint) == 64


def test_construction_command_has_vi_and_no_overwrite(tmp_path):
    args = argparse.Namespace(model="Qwen/Qwen3.5-2B", base_url="http://127.0.0.1:8000/v1",
                              chunk_size=3000, max_new_tokens=1536, max_extraction_chunks=500,
                              repetition_penalty=1.15, without_event_relations=True)
    command = construction_args(args, tmp_path / "input", tmp_path / "graph", "extract")
    assert command[command.index("--language") + 1] == "vi"
    assert "--overwrite" not in command
    assert "--resume-extraction" in command
    assert command[command.index("--max-extraction-chunks") + 1] == 500
    assert command[command.index("--repetition-penalty") + 1] == 1.15
    assert command.count("--without-event-relations") == 1


def test_no_event_construction_command_forwards_only_the_no_event_flag(tmp_path):
    args = argparse.Namespace(model="Qwen/Qwen3.5-2B", base_url="http://127.0.0.1:8000/v1",
                              chunk_size=3000, max_new_tokens=1536, max_extraction_chunks=1,
                              repetition_penalty=1.15, without_event_relations=False,
                              without_events=True)
    command = construction_args(args, tmp_path / "input", tmp_path / "graph", "extract")
    assert command.count("--without-events") == 1
    assert "--without-event-relations" not in command


def test_no_event_records_create_no_event_csv_nodes_or_edges(tmp_path):
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    (extraction / "hotpotqa_corpus_output.json").write_text(json.dumps({
        "id": "d1", "original_text": "Alice founded Acme.",
        "entity_relation_dict": [{"Head": "Alice", "Relation": "founded", "Tail": "Acme"}],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "csv"
    json2csv("hotpotqa_corpus", str(extraction), str(output), schema={}, custom=False)
    nodes = (output / "triple_nodes_hotpotqa_corpus_from_json_without_emb.csv").read_text(encoding="utf-8")
    edges = (output / "triple_edges_hotpotqa_corpus_from_json_without_emb.csv").read_text(encoding="utf-8")
    missing = (output / "missing_concepts_hotpotqa_corpus_from_json.csv").read_text(encoding="utf-8")
    assert ",event," not in nodes and "Event" not in missing
    assert "founded" in edges and "is participated by" not in edges


def test_extraction_progress_counts_durable_jsonl_records(tmp_path):
    extraction = tmp_path / "graph" / "kg_extraction"
    extraction.mkdir(parents=True)
    first = extraction / "Qwen_hotpotqa_corpus_output_1.json"
    second = extraction / "Qwen_hotpotqa_corpus_output_2.json"
    row = {"id": "doc", "original_text": "văn bản"}
    first.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, id="doc2")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(dict(row, id="doc3")) + "\n", encoding="utf-8")
    progress = inspect_extraction_progress(tmp_path / "graph", "hotpotqa_corpus")
    assert progress["completed_chunks"] == 3
    assert progress["files"] == {first.name: 2, second.name: 1}
    write_extraction_progress(tmp_path / "graph", progress)
    saved = json.loads((tmp_path / "graph" / "extraction_progress.json").read_text())
    assert saved["completed_chunks"] == 3 and saved["updated_at_utc"]


def test_vn_pilot_audit_uses_post_validation_records_and_log_warnings(tmp_path):
    root = tmp_path / "run"
    extraction = root / "experiment" / "graph" / "kg_extraction"
    extraction.mkdir(parents=True)
    records = [
        {"id": "a", "original_text": "A", "entity_relation_dict": [{"Head": "A"}],
         "event_entity_dict": [], "event_relation_dict": []},
        {"id": "b", "original_text": "B", "entity_relation_dict": [],
         "event_entity_dict": [], "event_relation_dict": []},
    ]
    (extraction / "Qwen_hotpotqa_corpus_output_1.json").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    (root / "extract.log").write_text(
        "Item 3 is a duplicate triple: {'Head': 'A'}\n"
        "Processed 1/2 chunks (this run: 1)\n"
        "Item 1 missing required keys: {'Tail'}. Problematic item: {}\n"
        "Processed 2/2 chunks (this run: 2)\n",
        encoding="utf-8",
    )
    loaded = load_records(root, 2)
    warnings = parse_log(root, 2)
    audit = make_audit(loaded, warnings)
    concentration = make_concentration(loaded, warnings)
    assert audit["valid_saved_items"] == 1
    assert audit["estimated_raw_candidates"] == 3
    assert audit["empty_chunks"] == 1
    assert concentration["duplicates_total"] == 1
    assert concentration["top_10_repetition_chunks"][0]["chunk"] == 1


def test_vietnamese_event_relation_guard_keeps_only_explicit_grounded_events():
    source = (
        "Raven ký hợp đồng với Activision trước khi các nhà phát triển rời Raven "
        "và thành lập Human Head Studios."
    )
    items = [
        {"Head": "Raven ký hợp đồng với Activision", "Relation": "trước",
         "Tail": "các nhà phát triển thành lập Human Head Studios"},
        {"Head": "Raven ký hợp đồng với Activision", "Relation": "bởi vì",
         "Tail": "các nhà phát triển thành lập Human Head Studios"},
        {"Head": "Raven Software", "Relation": "sau", "Tail": "Activision"},
        {"Head": "Raven ký hợp đồng với Activision", "Relation": "trước",
         "Tail": "Raven ký hợp đồng với Activision"},
    ]
    kept, dropped = sanitize_vietnamese_event_relations(items, source)
    assert kept == [items[0]]
    assert {item["reason"] for item in dropped} == {
        "relation_not_explicit_in_source", "endpoint_not_event_clause", "self_loop"
    }


def test_extraction_progress_rejects_corrupt_tail(tmp_path):
    extraction = tmp_path / "graph" / "kg_extraction"
    extraction.mkdir(parents=True)
    checkpoint = extraction / "Qwen_hotpotqa_corpus_output_1.json"
    checkpoint.write_text('{"id":"doc","original_text":"ok"}\n{"id":', encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt extraction checkpoint"):
        inspect_extraction_progress(tmp_path / "graph", "hotpotqa_corpus")


def test_triple_validator_drops_blank_endpoints_and_duplicates(monkeypatch):
    monkeypatch.setitem(sys.modules, "json_repair", types.SimpleNamespace(loads=json.loads))
    monkeypatch.setitem(sys.modules, "jsonschema", types.SimpleNamespace(validate=lambda **kwargs: None))
    path = ROOT / "atlas_rag/llm_generator/format/validate_json_output.py"
    spec = importlib.util.spec_from_file_location("validate_json_output_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = {"items": {"required": ["Head", "Relation", "Tail"], "properties": {
        key: {"type": "string", "minLength": 1} for key in ("Head", "Relation", "Tail")}}}
    response = json.dumps([
        {"Head": " A ", "Relation": " r ", "Tail": " B "},
        {"Head": "A", "Relation": "r", "Tail": "B"},
        {"Head": "A", "Relation": "r", "Tail": "   "},
        {"Head": "A", "Relation": "r", "Tail": None},
    ])
    assert module.fix_triple_extraction_response(response, schema=schema) == [
        {"Head": "A", "Relation": "r", "Tail": "B"}
    ]


def test_atlas_schema_requires_nonempty_strings():
    namespace = {}
    schema_path = ROOT / "atlas_rag/llm_generator/format/validate_json_schema.py"
    exec(schema_path.read_text(encoding="utf-8"), namespace)
    schema = namespace["ATLAS_SCHEMA"]
    for stage in schema.values():
        for value in stage["items"]["properties"].values():
            if value["type"] == "string":
                assert value["minLength"] == 1
            elif value["type"] == "array":
                assert value["minItems"] == 1
                assert value["items"]["minLength"] == 1


def test_vn_notebook_syntax_and_default_safe_phase():
    notebook = json.loads((ROOT / "colab/AutoSchemaKG_HotpotQA_VN.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    for expected in ("RUN_PHASE = 'prepare'", "data/hotpotqa_vi_1k/final", "scripts/run_hotpotqa_vn.py",
                     "intfloat/multilingual-e5-small", "requirements-colab.txt",
                     "EXPERIMENT_PROFILE = 'no_event_pilot'", "hotpotqa_vn_no_event_pilot",
                     "--without-event-relations", "--without-events", "--top-passages', '10'",
                     "'chunks': 109", "'penalty': 1.15",
                     "UPGRADE_CODE_FOR_RESUME = False"):
        assert expected in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"]
            ast.parse("".join(cell["source"]))
