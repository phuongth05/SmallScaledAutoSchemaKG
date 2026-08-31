# HotpotQA English 100-question pilot

This is a staged, resumable pilot. It is larger than the three-question smoke
test but remains exploratory; the AutoSchemaKG paper reports a random
1,000-question evaluation.

Use one persistent Drive directory and never reuse it after changing the
sample, model, chunking, retrieval settings, code or dependency lock.

## Protocol

- dataset: `hotpotqa/hotpot_qa`, `distractor`, `validation`
- sample: 100 uniform random questions, seed 42
- context: all ten HotpotQA distractor documents per question
- methods: dense, Entity-KG, Entity-Event-KG, Full-KG
- paper QA metrics: EM, token F1, document Recall@2 and Recall@5
- diagnostics: all-support, filtering errors/retries, error taxonomy, question
  type/difficulty breakdown and paired bootstrap intervals versus dense

## Commands

Run these with the Python 3.12 QA environment while the pinned local Qwen vLLM
server is available at `http://127.0.0.1:8000/v1`.

```bash
python -u scripts/run_hotpotqa_en.py --phase prepare \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_en_100q

python -u scripts/run_hotpotqa_en.py --phase extract \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_en_100q \
  --max-extraction-chunks 100
```

Rerun the identical `extract` command until `extraction_complete.json` exists.
The 100-chunk limit is per invocation and provides a clean checkpoint before a
Colab session limit or disconnect.

```bash
python -u scripts/run_hotpotqa_en.py --phase build \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_en_100q

python -u scripts/run_hotpotqa_en.py --phase package \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_en_100q

python -u scripts/run_hotpotqa_en.py --phase benchmark \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_en_100q \
  --filter-failure-policy error

python -u scripts/run_hotpotqa_en.py --phase report \
  --work-dir /content/drive/MyDrive/AutoSchemaKG/hotpotqa_en_100q
```

The benchmark checkpoints every successful `(method, question)`. Rerun the
same benchmark command after restarting vLLM to resume. The strict `error`
policy is recommended for reportable results; it never silently replaces a
failed graph filter with dense retrieval.

## Report artifacts

- `benchmark/summary.json`: aggregate runner output
- `benchmark/evaluation_100q.md`: report table
- `benchmark/evaluation_100q.json`: full breakdown, paired intervals and errors
- `benchmark/results/<method>/*.json`: auditable per-question records
- `construction.zip`: portable graph plus exact QA/corpus provenance

Keep the entire work directory or package it after evaluation. Embedding caches
are recomputable, but preserving them makes later analysis faster.
