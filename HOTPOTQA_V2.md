# HotpotQA v2 — closer to the paper, still a small-model adaptation

## What changed

The construction algorithm remains AutoSchemaKG. This upgrade replaces the
question-local lexical QA shortcut with global retrieval over the **pooled corpus
inside your construction ZIP**, using the authors' `HippoRAG2Retriever`.

| Component | v2 |
| --- | --- |
| Candidates | All corpus passages, never `sample.document_ids` or gold support labels |
| Embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, CPU, normalized |
| Edge search | FAISS exact inner-product index, top 30 semantic edges |
| Filtering | Original AutoSchemaKG fact-filter prompt, local Qwen, candidate-only validation |
| Propagation | Upstream HippoRAG2 passage/node seeds and directed NetworkX PageRank, alpha 0.9, passage weight 0.9 |
| QA evidence | Top 5 retrieved passages, not a lexical list of 60 triples |
| Comparison | Dense passage baseline, Entity-KG, Entity-Event-KG, Full-KG |
| Metrics | Answer EM/F1; document-title support recall and all-support hit at 2 and 5 |
| Persistence | Per-question atomic checkpoints, embedding batches cached, config/input/code fingerprint |

The three graph variants are **induced subgraphs of one constructed Full-KG**.
Passage nodes and their surviving source edges are retained in all variants.
Concepts are excluded from Entity/Entity-Event, so they cannot leak through the
index. Dense retrieval uses the same encoder, passages, answer prompt and reader.
No new LLM extraction is needed to create these ablations.

## Start with your existing ZIP

Open `colab/AutoSchemaKG_HotpotQA_PaperAligned.ipynb`. After these files have been
pushed to your GitHub, the notebook can be opened at:

[Open v2 in Colab](https://colab.research.google.com/github/phuongth05/SmallScaledAutoSchemaKG/blob/main/colab/AutoSchemaKG_HotpotQA_PaperAligned.ipynb)

1. Select GPU before uploading. The default server is local Qwen3.5-2B.
2. Mount Drive; set a new `RUN_ROOT` for this experiment.
3. Upload the construction/evaluated ZIP once. Do **not** upload `content.zip`.
4. Install two isolated environments: CPU embedding/QA client and GPU vLLM.
5. Start Qwen, then run the benchmark cell.
6. Inspect `summary.json`, per-question evidence and predictions; download the ZIP.

The old ZIP with 3 questions / 30 passages is supported. It remains a **smoke
test**, not a 100- or 1,000-question experiment merely because retrieval changed.
The original ZIP is read directly, never modified. Old QA scores in it are ignored.
The new archive includes an untouched copy of that ZIP and separate v2 outputs.

### CLI (inside the dedicated QA environment)

```bash
python scripts/run_hotpotqa_benchmark.py construction.zip --inspect-only

python -u scripts/run_hotpotqa_benchmark.py construction.zip \
  --output-dir outputs/hotpotqa_v2 \
  --variants dense entity entity_event full \
  --model Qwen/Qwen3.5-2B --base-url http://127.0.0.1:8000/v1 \
  --embedding-device cpu --top-edges 30 --top-passages 5 \
  --ppr-alpha 0.9 --passage-weight 0.9
```

For a CPU-only diagnostic, without starting the LLM:

```bash
python -u scripts/run_hotpotqa_benchmark.py construction.zip \
  --output-dir outputs/hotpotqa_v2_retrieval_diagnostic \
  --retrieval-only --no-filter-edges
```

This diagnostic reports retrieval metrics **only**, no EM/F1. Disabling the
LLM filter is a declared deviation and must not be presented as the full run.

## Next: prepare a larger, reproducible corpus

Keep the 3-question smoke run separate. First try 100 uniformly sampled validation
questions; increase to 1,000 after measuring your own extraction time and memory.
In the construction notebook, set `MAX_QUESTIONS=100`, `SAMPLING='random'`,
`SAMPLE_SEED=42`, `CONTEXT_MODE='all'` and choose new data/output folders.
Equivalent preparation command:

```bash
python -u scripts/prepare_hotpotqa.py \
  --config distractor --split validation --sampling random --seed 42 \
  --max-questions 100 --context-mode all --output-dir data/hotpotqa_v2_100
```

The API loader samples uniformly from the entire split using its reported row
count, fetches each needed page once, and records selected row/question IDs.
It does not shuffle just the first 100 rows. `--start-index` is for sequential
sampling only. A fixed seed makes this subset reproducible but does **not** make
it the exact subset selected by the paper. Pin/share the generated manifest.

Use the existing construction notebook/server to run `scripts/run_colab_v1.py`
against that new data directory. Include `dataset_metadata.json`,
`qa_manifest.json`, `hotpotqa_corpus.json`, extraction JSON and GraphML in its ZIP.
Then use v2 with a new `RUN_ROOT`. Construction is still the existing v1 process;
the new per-question resume mechanism applies to **benchmark inference**, not a
new guarantee of resumability for every graph-construction stage.

Do not use `--context-mode supporting`: it constructs an oracle-support-only
corpus. The v2 evaluator rejects archives explicitly marked this way.

## Checkpoints, interruptions and reproducibility

- `results/<method>/<question-hash>.json`: saved after each successful retrieval
  and answer. Contains original question, prediction, gold used for scoring,
  selected passage/document IDs, titles, actual evidence, selected facts, seeds,
  fallback status and metrics. Gold is never passed to retrieval or generation.
- `summary.json`: completed/expected counts and averages on completed rows.
  Compare final methods only when `complete` is true for every method.
- `run_config.json`: input fingerprint, settings, code hashes and dependency
  versions. Changed input/model/settings/code/dependencies require a new folder.
- `embedding_cache/`: normalized vectors in non-pickle `.npy` batches.
- `last_error.json`: last failure, if any; historical diagnostic, not final status.
- Notebook Drive folder: input ZIP, code/model revisions, dependency locks,
  benchmark outputs and vLLM log. Model weights stay in the runtime cache.

After disconnect: reconnect, rerun setup/server cells, then the **same benchmark
command**. Successful question checkpoints are skipped; an interrupted question
is rerun. If `/content` was reset, Drive is still the source of saved work. Without
Drive or a downloaded archive, ephemeral runtime data is not durable.

The notebook pins Git and Hugging Face model commits in `revisions.json`, and
saves `uv pip freeze` locks. CLI users should pass `--model-revision` and
`--embedding-revision` for equivalent provenance. The tokenizer revision must
match the weights/revision served by vLLM. Do not reuse an unrelated server on
the same port. If changing GPU makes a saved wheel lock incompatible, create a
new run environment rather than silently mixing results.
Run only one benchmark process per output directory at a time.

The context guard budgets using the reader's tokenizer. If 30 edge candidates
do not fit, the lowest-ranked candidates are removed and the count is recorded.
Retrieved passages are evenly token-truncated when needed; the exact snippets
shown to the reader are saved. A generation stopped by its token limit is an
error, not a silently scored partial answer. Empty valid fact selection invokes
the upstream dense-passage fallback and is recorded. Malformed/invented facts
raise an error; completed questions are preserved for retry.
MiniLM also has its own maximum input length; embedding long passages may
truncate them even when the reader can accept more tokens.

## How to interpret results

Metrics are 0–1, multiply by 100 for percentage tables.

- `em` / `f1`: normalized answer scores, including HotpotQA's special handling
  of yes/no/noanswer. This is **not** joint answer/supporting-fact F1.
- `support_recall@k`: fraction of gold supporting document titles found in the
  first k retrieved graph passages. Multiple chunks from one title count once.
- `all_support@k`: whether every gold supporting title is present in top k.
  This is document-level retrieval evaluation, not sentence-level support F1.

Inspect cases where recall is low before blaming the reader. When recall is
high but EM/F1 is low, inspect input truncation, reader output and answer format.
Compare Full-KG with Entity-Event-KG to measure whether the concepts help **this
retriever**; do not assume more concepts imply better quality. Tiny samples or
partial runs do not support a claim of improvement.

## Remaining differences from the paper

Sources: [AutoSchemaKG paper](https://arxiv.org/abs/2505.23628),
[authors' code](https://github.com/HKUST-KnowComp/AutoSchemaKG),
[MiniLM model card](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1).

Still different: Qwen3.5-2B instead of the paper's Llama construction/reader
models; compact cosine MiniLM instead of the original benchmark's larger
embedding setup; exact FAISS rather than HNSW; strict candidate matching rather
than embedding-remapping filtered triples; short-answer prompting and limited
context; smaller/differently sampled corpus. Construction keeps the earlier
3000-**character** chunk setting and small batches, not the paper's 1024-token
and batching settings. These are not equivalent units.

The paper's multi-hop experiment constructs graphs from the corresponding QA
context corpora. It does **not** require rebuilding the billion-scale ATLAS
corpora to attempt a HotpotQA reproduction. Our global search is over the pooled
sampled contexts, **not all Wikipedia**. We have not added ToG/HippoRAG1,
MuSiQue/2Wiki, triple-accuracy judges, schema-typing evaluation or FELM/MMLU.

This upgrade is suitable for studying the effect of event/concept graph
structure under constrained local-LLM resources. It must not be reported as
reproducing the paper's published EM/F1.

## Validation

```bash
python -m pytest tests/test_hotpotqa_v2.py -q
python scripts/check_colab_v1.py
python scripts/check_hotpotqa.py
python scripts/check_hotpotqa_inference.py
```

Tests use deterministic fixture embeddings and mocked LLM answers, with real
FAISS and upstream PageRank. They check ablations, multi-chunk provenance,
candidate filtering, empty-filter fallback, scoring, interrupted-run resume,
configuration guards and notebook syntax. These are correctness checks, not
evidence of Qwen answer quality or a successful GPU/Colab execution.
