"""Regression for the reported Qwen 4/5-element fact bug; no GPU/network calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hotpotqa_benchmark as runner
import colab_v2_utils as utils
from test_hotpotqa_v2 import FixtureEncoder, args_for, graph, write_bundle
ReaderClass = runner.LocalReader

BAD_RESPONSE = json.dumps({"fact": [
    ["Scott Derrickson", "is", "American"],
    ["Scott Derrickson", "is", "born", "July 16, 1966"],
    ["Ed Wood", "directed by", "Tim Burton", "involves", "Patricia Arquette"],
]})
FACTS = [["Scott Derrickson", "is", "American"], ["Ed Wood", "is", "American"]]


def make_reader(responses, policy="error"):
    reader = ReaderClass.__new__(ReaderClass)
    reader.args = NS(context_length=4096, max_filter_tokens=128, filter_max_attempts=2,
                     filter_failure_policy=policy, language="en")
    reader.token_count = lambda messages: 80
    iterator = iter(responses)
    reader.messages = []
    def generate(messages, budget):
        reader.messages.append(messages)
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        reader.last_usage = {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90}
        return value, 80
    reader.generate = generate
    return reader


@pytest.mark.parametrize("value", ["http://127.0.0.1:8000/v1", " http://127.0.0.1:8000/v1/ ",
    "[http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1)"])
def test_url_normalization(value):
    assert utils.normalize_base_url(value) == "http://127.0.0.1:8000/v1"


@pytest.mark.parametrize("value", ["http://127.0.0.1:8192v1", "file:///v1", "http://a:bad/v1",
    "http://a/v1?q=1", "http://user:secret@a/v1", "http://a:8000", "http://a b/v1"])
def test_bad_url_rejected(value):
    with pytest.raises(ValueError, match="base-url"):
        utils.normalize_base_url(value)


@pytest.mark.parametrize("text", [BAD_RESPONSE, '{"selected_ids": ["0"]}', '{"selected_ids": [true]}',
    '{"selected_ids": [-1]}', '{"selected_ids": [2]}', '{"selected_ids": [[0]]}',
    '{"selected_ids": [0.0]}', '{"selected_ids": [0], "fact": []}', 'Not JSON'])
def test_strict_ids_reject_wrong_shapes_and_invented_values(text):
    with pytest.raises(ValueError):
        utils.parse_selected_ids(text, 2)


def test_valid_ids_deduplicated_and_fenced_json_supported():
    assert utils.parse_selected_ids('```json\n{"selected_ids": [1,0,1]}\n```', 2) == [1, 0]
    assert utils.parse_selected_ids('{"selected_ids": []}', 2) == []


def test_reported_malformed_facts_retry_then_select_exact_candidates():
    reader = make_reader([BAD_RESPONSE, '{"selected_ids": [1, 0]}'])
    selected, info = reader.filter_facts("Same nationality?", FACTS)
    assert selected == [FACTS[1], FACTS[0]]
    assert info["attempts"][0]["raw_response"] == BAD_RESPONSE
    assert len(info["attempts"]) == 2 and "error" in info["attempts"][0]
    assert info["usage"]["total_tokens"] == 180
    assert info["input_tokens"] == 160 and not info["fallback_due_to_error"]
    assert reader.messages[0][-1] == reader.messages[1][-1]  # Stable candidate IDs.
    assert reader.messages[0][0] != reader.messages[1][0]  # Explicit retry instruction.


def test_exhausted_invalid_ids_stop_in_strict_mode_with_raw_response():
    reader = make_reader([BAD_RESPONSE, BAD_RESPONSE])
    with pytest.raises(utils.FilterSelectionError) as error:
        reader.filter_facts("Question", FACTS)
    assert error.value.diagnostics["attempts"][1]["raw_response"] == BAD_RESPONSE
    assert not error.value.diagnostics["fallback_due_to_error"]


def test_explicit_dense_fallback_does_not_use_malformed_triples(graph, monkeypatch):
    import networkx as nx
    from hotpotqa_benchmark import index_graph, make_hipporag2
    reader = make_reader([BAD_RESPONSE, BAD_RESPONSE], "dense")
    encoder = FixtureEncoder()
    retriever = make_hipporag2(index_graph(graph, encoder), encoder, reader)
    monkeypatch.setattr(nx, "pagerank", lambda *a, **k: pytest.fail("Error fallback must use dense passages"))
    passages, _ = retriever.retrieve("Question", topN=5)
    assert len(passages) == 5
    assert retriever.trace["selected_facts"] == []
    assert retriever.trace["dense_fallback_reason"] == "invalid_filter_output_exhausted"
    assert retriever.trace["filter"]["fallback_due_to_error"] is True


def test_valid_empty_selection_is_not_format_error():
    reader = make_reader(['{"selected_ids": []}'], "dense")
    selected, info = reader.filter_facts("Question", FACTS)
    assert selected == [] and len(info["attempts"]) == 1
    assert not info["fallback_due_to_error"]


def test_incomplete_generation_retried_but_connection_error_not_swallowed():
    reader = make_reader([utils.IncompleteGenerationError("length"), '{"selected_ids": [0]}'])
    assert reader.filter_facts("Question", FACTS)[0] == FACTS[:1]
    reader = make_reader([ConnectionError("server down")], "dense")
    with pytest.raises(ConnectionError, match="server down"):
        reader.filter_facts("Question", FACTS)
    assert len(reader.messages) == 1


def test_v2_checkpoints_persist_explicit_fallback_and_resume(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run")
    args.variants, args.no_filter_edges = ["entity"], False
    args.filter_failure_policy = "dense"
    args.base_url = "[local](http://127.0.0.1:8000/v1)"
    monkeypatch.setattr(runner, "CachedEncoder", FixtureEncoder)
    monkeypatch.setattr(runner, "LocalReader", lambda args: make_reader([BAD_RESPONSE, BAD_RESPONSE], "dense"))
    result = runner.run(args)
    counts = result["methods"]["entity"]["diagnostics"]
    assert counts["filter_error_fallback_questions"] == 1
    assert counts["filter_retry_calls"] == 1 and counts["invalid_filter_attempts"] == 2
    monkeypatch.setattr(runner, "LocalReader", lambda args: pytest.fail("No LLM on complete resume"))
    assert runner.run(args) == result
    args.filter_failure_policy = "error"
    with pytest.raises(ValueError, match="another graph/config"):
        runner.run(args)


def test_strict_failure_saves_raw_diagnostics_without_checkpoint(tmp_path, graph, monkeypatch):
    args = args_for(write_bundle(tmp_path / "bundle.zip", graph), tmp_path / "run")
    args.variants, args.no_filter_edges = ["entity"], False
    monkeypatch.setattr(runner, "CachedEncoder", FixtureEncoder)
    monkeypatch.setattr(runner, "LocalReader", lambda args: make_reader([BAD_RESPONSE, BAD_RESPONSE]))
    with pytest.raises(utils.FilterSelectionError):
        runner.run(args)
    error = json.loads((tmp_path / "run/last_error.json").read_text(encoding="utf-8"))
    assert error["filter_diagnostics"]["attempts"][0]["raw_response"] == BAD_RESPONSE
    assert not list((tmp_path / "run").rglob("results/*.json"))


def test_notebook_log_stream_preserves_actual_child_error(tmp_path, capsys):
    log = tmp_path / "error.log"
    with pytest.raises(RuntimeError, match="actual root cause"):
        utils.run_logged([sys.executable, "-c", "print('first output'); raise ValueError('actual root cause')"], log)
    assert "actual root cause" in log.read_text(encoding="utf-8")
    assert "first output" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        utils.run_logged([sys.executable, "-c", "print('not run')"], log)


def test_generation_keeps_truncated_raw_response():
    reader = runner.LocalReader.__new__(runner.LocalReader)
    reader.args = NS(context_length=4096, model="fixture")
    reader.token_count = lambda m: 80
    reader.client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(
        choices=[NS(finish_reason="length", message=NS(content='{"selected_ids": ['))],
        usage=NS(prompt_tokens=80, completion_tokens=10, total_tokens=90)))))
    with pytest.raises(utils.IncompleteGenerationError):
        reader.generate([], 128)
    assert reader.last_generation["raw_response"] == '{"selected_ids": ['


def test_notebook_new_run_root_and_explicit_filter_policy():
    notebook = json.loads((ROOT / "colab/AutoSchemaKG_HotpotQA_PaperAligned.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "hotpotqa_v2_id_filter_smoke" in source
    assert "--filter-failure-policy" in source and "run_logged(command" in source
    assert "--no-filter-edges" not in source
