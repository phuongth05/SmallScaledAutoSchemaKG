# Optional remote vLLM backend

The HotpotQA-VN workflow can send only LLM requests to an authenticated
OpenAI-compatible vLLM server. Google Drive remains authoritative for inputs,
graphs, checkpoints, benchmark results, and provenance. `local` remains the
default backend.

## RunPod RTX 4090 server

Use the same pinned model and revision as the Colab experiment:

```bash
git clone --branch codex/research-concept-retrieval https://github.com/phuongth05/SmallScaledAutoSchemaKG.git
cd SmallScaledAutoSchemaKG
python -m pip install --upgrade pip
python -m pip install --pre vllm
export VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export MODEL_ID='Qwen/Qwen3.5-2B'
export MODEL_REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
export MAX_MODEL_LEN=4096
export GPU_MEMORY_UTILIZATION=0.90
bash scripts/start_remote_vllm.sh
```

Keep `VLLM_API_KEY` in a secret manager or RunPod secret. Do not paste it into
Git, a saved notebook, Drive metadata, or logs. Exposing raw vLLM publicly is
not ideal security; require the API key and expose only port 8000 through the
minimum necessary proxy/firewall surface. The script does not print the key.

From a different shell, verify the endpoint:

```bash
export LLM_BACKEND=remote
export REMOTE_LLM_BASE_URL='https://YOUR-POD-ID-8000.proxy.runpod.net/v1'
export REMOTE_LLM_API_KEY='THE-SAME-SECRET'
python scripts/check_remote_llm.py --model Qwen/Qwen3.5-2B
```

## Colab remote mode

Set runtime-only environment variables. `getpass` prevents the key from being
echoed into notebook output:

```python
import getpass
import os

os.environ["LLM_BACKEND"] = "remote"
os.environ["REMOTE_LLM_BASE_URL"] = input("Remote vLLM /v1 URL: ").strip()
os.environ["REMOTE_LLM_API_KEY"] = getpass.getpass("Remote vLLM API key: ")
```

Then run the existing phase with the same source and work directory. The
wrapper health-checks `/v1/models` before any experiment write, passes Bearer
authentication to extraction and benchmark clients, and resumes the existing
checkpoint:

```bash
python -X utf8 -u scripts/run_hotpotqa_vn.py \
  --phase extract \
  --llm-backend remote \
  --source-dir /content/Prepare-data-HotpotQA-VN/data/hotpotqa_vi_1k/final \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_vn_ab_entity_event_v3/experiment \
  --max-questions 1000 --sampling random --seed 42 \
  --model Qwen/Qwen3.5-2B \
  --model-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --chunk-size 3000 --max-new-tokens 1536 --repetition-penalty 1.15 \
  --context-length 4096 --without-event-relations \
  --max-extraction-chunks 1
```

Use `--max-extraction-chunks 1` for the first remote test. It adds at most one
new durable chunk to an existing partial extraction and does not redo completed
chunks. If `vn_extraction_complete.json` exists, extraction remains skipped;
use only the read-only health checker instead of creating another experiment.
After verifying the response/checkpoint, raise the limit to 5 and then continue
with the existing planned limit.

Network resets, timeouts, and HTTP 429/502/503/504 receive bounded exponential
backoff. Authentication and deterministic request failures are not repeatedly
retried. A final request failure exits before the chunk result is written, so
the existing resume scan remains authoritative.

## Switch back to local mode

For a shell-based run:

```bash
export LLM_BACKEND=local
unset REMOTE_LLM_BASE_URL
unset REMOTE_LLM_API_KEY
```

For the Colab notebook:

```python
import os

os.environ["LLM_BACKEND"] = "local"
os.environ.pop("REMOTE_LLM_BASE_URL", None)
os.environ.pop("REMOTE_LLM_API_KEY", None)
```

Start the existing local Colab vLLM cell and rerun the unchanged phase. The
default endpoint remains `http://127.0.0.1:8000/v1`.

For an existing Drive run pinned to an older code commit, set the notebook's
existing `UPGRADE_CODE_FOR_RESUME=True` once after this change is available on
the branch. The notebook records the previous commit in
`code_upgrade_history`; set the flag back to `False` immediately afterward.
This uses the repository's existing code-upgrade mechanism and does not alter
the construction configuration or completed chunk offset.

## Concurrency limitation

Client-side extraction concurrency remains `1`. The current durable resume
offset is defined in units that assume `batch_size_triple=1`; changing that
batch size after a partial run can skip chunks, while concurrent checkpoint
writes are not independently coordinated. Consequently this integration does
not expose `LLM_CONCURRENCY>1`. vLLM may use its own internal scheduling, but a
single pipeline run submits one chunk at a time. A future concurrency change
requires a per-chunk-ID checkpoint redesign and is intentionally out of scope.

## Rollback

1. Stop the Colab phase after its current atomic checkpoint.
2. Set `LLM_BACKEND=local` and remove the two remote variables as shown above.
3. Start the local vLLM cell.
4. Rerun the exact same phase and work directory.

No dataset, graph, checkpoint, completion marker, or provenance file needs to
be moved, renamed, deleted, or regenerated. Endpoint URLs, backend selection,
pod IDs, and credentials are excluded from experiment-semantic configuration.
