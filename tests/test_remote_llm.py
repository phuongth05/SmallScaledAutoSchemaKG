"""Offline transport tests; no network, GPU, model, or dataset required."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import llm_endpoint
import run_colab_v1
import run_hotpotqa_benchmark as benchmark
import run_hotpotqa_vn


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"data": [{"id": "Qwen/Qwen3.5-2B"}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_local_defaults_are_unchanged(monkeypatch):
    for name in ("LLM_BACKEND", "REMOTE_LLM_BASE_URL", "REMOTE_LLM_API_KEY",
                 "LOCAL_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    connection = llm_endpoint.resolve_llm_connection()
    assert connection.backend == "local"
    assert connection.base_url == "http://127.0.0.1:8000/v1"
    assert connection.api_key == "EMPTY"


def test_remote_health_uses_bearer_auth_and_expected_model(monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_BASE_URL", "https://pod.example/v1/")
    monkeypatch.setenv("REMOTE_LLM_API_KEY", "do-not-persist")
    connection = llm_endpoint.resolve_llm_connection("remote")
    session = FakeSession(FakeResponse())
    assert llm_endpoint.check_llm_health(
        connection, "Qwen/Qwen3.5-2B", session=session
    ) == ["Qwen/Qwen3.5-2B"]
    assert session.calls == [("https://pod.example/v1/models", {
        "headers": {"Authorization": "Bearer do-not-persist"}, "timeout": 15
    })]
    assert "do-not-persist" not in repr(connection)


def test_remote_auth_failure_does_not_modify_state(tmp_path):
    connection = llm_endpoint.LLMConnection(
        "remote", "https://pod.example/v1", "wrong-secret"
    )
    with pytest.raises(RuntimeError, match="authentication failed"):
        llm_endpoint.check_llm_health(
            connection, "Qwen/Qwen3.5-2B",
            session=FakeSession(FakeResponse(status=401)),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status, expected", [(429, True), (503, True), (401, False), (400, False)])
def test_only_transient_http_failures_are_retryable(status, expected):
    error = RuntimeError("request failed")
    error.status_code = status
    assert llm_endpoint.is_transient_llm_error(error) is expected


def test_remote_command_has_no_secret_and_local_command_is_stable(tmp_path):
    base = dict(model="Qwen/Qwen3.5-2B", base_url="http://127.0.0.1:8000/v1",
                chunk_size=3000, max_new_tokens=1536, max_extraction_chunks=1,
                repetition_penalty=1.15, without_event_relations=True)
    local = run_hotpotqa_vn.construction_args(
        argparse.Namespace(**base, llm_backend="local"), tmp_path / "in", tmp_path / "out", "extract"
    )
    remote = run_hotpotqa_vn.construction_args(
        argparse.Namespace(**base, llm_backend="remote"), tmp_path / "in", tmp_path / "out", "extract"
    )
    assert "--llm-backend" not in local
    assert remote[-2:] != ["--llm-backend", "remote"]  # ablation flag remains last
    assert remote[remote.index("--llm-backend") + 1] == "remote"
    assert "REMOTE_LLM_API_KEY" not in " ".join(map(str, remote))


def test_endpoint_switch_is_semantically_compatible():
    current = {"settings": {"model": "m"}, "code": {"run_hotpotqa_benchmark.py": "new"}}
    old = json.loads(json.dumps(current))
    old["code"]["run_hotpotqa_benchmark.py"] = next(
        iter(benchmark.TRANSPORT_COMPATIBLE_SCRIPT_DIGESTS)
    )
    assert benchmark.transport_compatible_config(old, current)
    changed = json.loads(json.dumps(old))
    changed["settings"]["model"] = "another-model"
    assert not benchmark.transport_compatible_config(changed, current)


def test_benchmark_endpoint_switch_keeps_same_semantic_settings():
    common = dict(source=Path("graph"), output_dir=Path("benchmark"), model="model",
                  variants=["full"], inspect_only=False)
    local = argparse.Namespace(**common, base_url="http://127.0.0.1:8000/v1",
                               llm_backend="local")
    remote = argparse.Namespace(**common, base_url="https://pod.example/v1",
                                llm_backend="remote")
    assert benchmark.benchmark_settings(local) == benchmark.benchmark_settings(remote)


def colab_args(data_dir, output_dir):
    return argparse.Namespace(
        data_dir=data_dir, output_dir=output_dir, filename_pattern="hotpotqa_corpus",
        overwrite=False, resume_extraction=True, max_extraction_chunks=1,
        repetition_penalty=1.15, phase="extract", without_concepts=False,
        llm_backend="remote", base_url="http://127.0.0.1:8000/v1", api_key=None,
        model="Qwen/Qwen3.5-2B",
    )


def test_failed_health_check_precedes_checkpoint_write(tmp_path, monkeypatch):
    data = tmp_path / "input"
    data.mkdir()
    (data / "hotpotqa_corpus.json").write_text("[]", encoding="utf-8")
    output = tmp_path / "graph"
    monkeypatch.setattr(run_colab_v1, "parse_args", lambda: colab_args(data, output))
    monkeypatch.setattr(llm_endpoint, "resolve_llm_connection", lambda *a: object())
    monkeypatch.setattr(
        llm_endpoint, "check_llm_health",
        lambda *a: (_ for _ in ()).throw(RuntimeError("health failed")),
    )
    with pytest.raises(RuntimeError, match="health failed"):
        run_colab_v1.main()
    assert not output.exists()


def test_transient_extraction_failure_never_marks_chunk_complete(tmp_path, monkeypatch):
    data = tmp_path / "input"
    data.mkdir()
    (data / "hotpotqa_corpus.json").write_text("[]", encoding="utf-8")
    output = tmp_path / "graph"
    connection = llm_endpoint.LLMConnection("remote", "https://pod.example/v1", "secret")
    monkeypatch.setattr(run_colab_v1, "parse_args", lambda: colab_args(data, output))
    monkeypatch.setattr(llm_endpoint, "resolve_llm_connection", lambda *a: connection)
    monkeypatch.setattr(llm_endpoint, "check_llm_health", lambda *a: ["Qwen/Qwen3.5-2B"])
    extractor = SimpleNamespace(
        run_extraction=lambda: (_ for _ in ()).throw(requests.ConnectionError("reset"))
    )
    monkeypatch.setattr(run_colab_v1, "make_extractor", lambda args: extractor)
    with pytest.raises(requests.ConnectionError):
        run_colab_v1.main()
    progress = json.loads((output / "extraction_progress.json").read_text(encoding="utf-8"))
    assert progress["completed_chunks"] == 0
    assert not (output / "vn_extraction_complete.json").exists()


def test_completed_extraction_marker_skips_health_and_extraction(tmp_path, monkeypatch):
    work = tmp_path / "experiment"
    data = work / "input"
    graph = work / "graph"
    provenance = graph / "provenance"
    data.mkdir(parents=True)
    provenance.mkdir(parents=True)
    payloads = {
        "hotpotqa_corpus.json": "[]",
        "qa_manifest.json": "[]",
        "dataset_metadata.json": json.dumps({
            "language": "vi", "retrieval_ground_truth": "qrels_document_ids"
        }),
    }
    for name, value in payloads.items():
        (data / name).write_text(value, encoding="utf-8")
        (provenance / name).write_text(value, encoding="utf-8")
    (graph / "vn_extraction_complete.json").write_text('{"complete": true}', encoding="utf-8")
    config = {
        "model": "Qwen/Qwen3.5-2B", "model_revision": "revision", "language": "vi",
        "chunk_size": 3000, "max_new_tokens": 1536, "repetition_penalty": 1.15,
        "include_event_relations": False,
        "corpus_sha256": hashlib.sha256((data / "hotpotqa_corpus.json").read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(
            (ROOT / "atlas_rag/llm_generator/prompt/vietnamese.py").read_bytes()
        ).hexdigest(),
    }
    (graph / "vn_construction_config.json").write_text(json.dumps(config), encoding="utf-8")
    args = argparse.Namespace(
        phase="extract", work_dir=work, source_dir=None, max_questions=1000,
        sampling="random", seed=42, model="Qwen/Qwen3.5-2B", model_revision="revision",
        base_url="http://127.0.0.1:8000/v1", llm_backend="remote", chunk_size=3000,
        max_new_tokens=1536, repetition_penalty=1.15, without_event_relations=True,
        max_extraction_chunks=1, retrieval_only=False, no_filter_edges=False,
        variants=["full"], embedding_model="unused", embedding_revision=None,
        context_length=4096,
    )
    monkeypatch.setattr(run_hotpotqa_vn, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_hotpotqa_vn, "check_llm_health",
        lambda *a: pytest.fail("completed extraction must not contact the LLM"),
    )
    monkeypatch.setattr(
        run_hotpotqa_vn, "call_script",
        lambda *a: pytest.fail("completed extraction must remain skipped"),
    )
    run_hotpotqa_vn.main()
