# Research: concept selection and retrieval with a small local LLM

Branch: `codex/research-concept-retrieval`.

This is an **experiment setup**, not a claim of improvement or an exact reproduction
of AutoSchemaKG. It builds on the existing v2/Vietnamese work without replacing it.
Original construction ZIPs are read-only. No extraction, training, deployment or
GitHub push is triggered by the research runner.

## Research questions and controls

1. Does competition from concept edges reduce factual-edge recall in candidate search?
2. Can candidate quotas or factual-only seeds improve evidence retrieval, while
   retaining concepts during PageRank propagation?
3. Can post-hoc concept weighting/pruning reduce index size without hurting QA?

The existing 3-question / 30-passage graph is **only a smoke test**. Do not optimize
thresholds against those three answers or report its bootstrap intervals as evidence.

Freeze on dev: corpus/manifest, constructor output, reader and embedding revisions,
context budget, suite, random seed and parameters. Then evaluate chosen arms on test.
Only oracle diagnostics may inspect gold before retrieval. Normal methods receive
question text only; `document_ids` and qrels never restrict the candidate corpus.

## Implemented matrix

All definitions live in [research/experiments.json](research/experiments.json).

| Arm | Intervention |
| --- | --- |
| `bm25` | Okapi BM25, k1=1.5, b=0.75, positive log IDF; Unicode word/syllable tokens, no stopwords |
| `dense` | Same embedding model/corpus as graph arms |
| `entity` | Entity + passage induced subgraph |
| `entity_event` | Entity + event + passage |
| `full` | Full graph, unchanged retrieval baseline |
| `full_no_filter` | QA-only control: Full without LLM edge filtering |
| `quota_20_10` | 20 factual + 10 concept candidate edges; backfill if a group is undersupplied |
| `factual_seeds` | Search only factual candidates; concepts remain available during propagation |
| `concept_weight_025` | Multiply concept-edge transition weights by 0.25; other weights unchanged |
| `concept_cap_1`, `concept_cap_3` | Retain at most 1/3 distinct concepts per non-concept node |
| `random_matched_cap_1` | Remove the same number of concept edges as cap-1, chosen randomly with fixed seed |
| `oracle` | QA diagnostic: gold supporting documents supplied to reader; **not a competitive retriever** |
| `no_context` | QA diagnostic: same reader/prompt with no passages |

Cap selection currently prefers concepts connected to fewer distinct non-concept
neighbors, with stable ID tie-breaking. **Rarity is a hypothesis, not a quality label.**
Forward/reverse attachments to a retained concept stay together. Random control
matches removed edge count, not node count or degree distribution. Concept-concept
edges, if present, are not covered by the per-nonconcept-node cap.

Every variant retains the same passages. Only isolated concept nodes are removed.
Pruning does **not** reclaim already-spent graph-construction tokens/time. This first
matrix does not implement semantic concept merging, query-dependent weights,
selective re-extraction or a stronger-model verifier; those are follow-up experiments.

## Two execution stages

V2.1 updates the shared reader to candidate-ID filtering. Research runs default to
`--filter-max-attempts 2 --filter-failure-policy error`; use a NEW output root after
upgrading. Optional `--filter-failure-policy dense` is a distinct declared protocol:
only exhausted invalid/incomplete filter outputs trigger per-question dense fallback.
Network failures still stop. Reports count filter-error fallbacks and actual retry calls;
raw attempts and per-call token usage are retained. Do not combine old text-filter
results with these ID-filter results.

- `diagnostic`: CPU retrieval, **all LLM filters disabled**, no reader and no EM/F1.
  BM25 needs no model; dense/graph download an embedding model if not cached.
- `qa`: same retrieval plus local Qwen fact filtering (graph arms) and short-answer
  generation. `full_no_filter` isolates filter effects. Requires a running compatible
  OpenAI-style local server; use the research notebook's isolated vLLM environment.

Defaults: Qwen/Qwen3.5-2B, top-30 edges, PPR alpha=0.9, passage seed weight=0.9,
retrieve top-10 passages, answer with first 5, 4096-token context. EN uses
MiniLM; VI automatically uses `intfloat/multilingual-e5-small` with E5 prefixes.
The prepared VN corpus can be evaluated by BM25/dense **before building a graph**.
These are adapted small-model settings, not the paper's reader/model configuration.

## Local CLI / commands inside the Colab QA environment

Use the existing `requirements-hotpotqa-v2.txt` in an isolated CPU environment.
On this Windows workspace the already-installed test interpreter is
`outputs/qa-v2-test-env/Scripts/python.exe`; below `python` means that environment,
not necessarily the system Python. No new dependency is needed for BM25.

First save a plan and a stratified human-review template; no model is loaded:

```bash
python scripts/run_research_experiments.py autoschemakg_hotpotqa_evaluated.zip --output-dir outputs/research_smoke --audit-per-type 25
```

Then run CPU diagnostics (all 11 diagnostic arms by default):

```bash
python -u scripts/run_research_experiments.py autoschemakg_hotpotqa_evaluated.zip --output-dir outputs/research_smoke --execute
python scripts/report_research_experiments.py outputs/research_smoke
```

Limit a run to chosen arms without changing the frozen suite:

```bash
python -u scripts/run_research_experiments.py autoschemakg_hotpotqa_evaluated.zip --output-dir outputs/research_smoke --arms dense full quota_20_10 factual_seeds --execute
```

After starting Qwen with matching weights/revision, run the QA smoke baselines:

```bash
python -u scripts/run_research_experiments.py autoschemakg_hotpotqa_evaluated.zip --output-dir outputs/research_smoke --stage qa --arms bm25 dense entity entity_event full full_no_filter oracle no_context --execute
python scripts/report_research_experiments.py outputs/research_smoke --stage qa
```

For a larger bundle, create a NEW output directory. Use `--split dev --dev-size 100`
on every command; reserve the remaining questions for `--split test --allow-test`.
For 1,000 queries this means 100 dev / 900 test; for <=100 queries there is no test
under the default dev-size. The runner refuses empty splits. Test execution requires
the explicit flag; it does not automatically choose a winning arm.

VN baselines on prepared data, without a KG or Qwen server:

```bash
python -u scripts/run_research_experiments.py outputs/hotpotqa_vn/input --output-dir outputs/research_vn_baselines --stage diagnostic --split dev --arms bm25 dense --execute
```

Use the actual input folder created by `run_hotpotqa_vn.py --phase prepare`.
`max_questions` reduces evaluated queries, **not** the 9,822-document VN corpus.
After a VN graph is constructed, use its bundle with a NEW research output root.

## Colab

Open [colab/AutoSchemaKG_Research.ipynb](colab/AutoSchemaKG_Research.ipynb).
Its clone cell targets the research branch, **which must first be pushed to GitHub**.
Creating a local branch does not make that link runnable remotely.

1. Keep `STAGE='diagnostic'`, `EXECUTE=False`, `SPLIT='smoke'` initially.
2. Mount Drive, upload the construction ZIP, install the isolated CPU client.
3. Inspect the plan; set `EXECUTE=True` to run selected arms.
4. For full QA, choose a GPU runtime and `STAGE='qa'`; rerun setup and server cells.
5. Read the generated report and optionally download a result ZIP.

Notebook saves Git/model revisions and CPU/vLLM dependency locks. CLI users should
pass `--model-revision` and `--embedding-revision` for equivalent pinning. The server
must serve those actual weights: its advertised model ID alone does not prove revision.
GPU/vLLM compatibility still needs a real Colab test; offline tests do not validate it.
The dependency freeze records resolved versions, not a platform-independent wheel lock.

## Outputs, resume, and safety

```text
research_root/
  protocol.json                  immutable suite, source/config/code/dependency hashes
  splits.json                    fixed smoke/dev/test question IDs and overlap warning
  plan_<stage>_<split>.json       selected arms; plan-only by default
  audit_template.json            human labels; never overwritten by a rerun
  embedding_cache/               shared, content-addressed .npy batches
  diagnostic|qa/smoke|dev|test/<arm>/
    config.json                  exact arm and protocol ID
    sessions.json                per-session setup timing, graph sizes, pending count
    results/<question-hash>.json evidence, answer, trace, token usage, elapsed times
    summary.json                 expected/completed counts and means
    last_error.json              historical failure, if any
  report_<stage>_<split>.json    paired bootstrap, breakdowns, cost and exclusions
  report_<stage>_<split>.md      compact overview
```

Rerun identical arguments to skip completed questions. Changing code, dependency
versions, model, dataset, suite or parameters requires a NEW output directory.
Selecting other arms or switching stage/split does not change the protocol.
Embedding cache is shared: index/setup timings are warm/cache-dependent, not a fair
cold-build speed comparison. GPU peak memory is not measured by this runner.

Only one writer per output root. A `.research.lock` prevents simultaneous runs.
After a killed runtime, confirm no process is still writing before manually removing
that one lock file. No script deletes it automatically on behalf of another process.
Drive/checkpoints survive a runtime reset; `/content` alone does not. The notebook ZIP
excludes the recomputable embedding cache; original input and experiment records stay.

## Metrics and interpretation

- Support-document recall and all-support hit at 2/5/10. Passage slots, not distinct
  documents, consume top-k. VN uses qrels document IDs; EN uses supporting titles.
- EM/F1 only in QA mode. VN F1 is whitespace/syllable overlap with accents retained,
  **not a word-segmented Vietnamese metric**. Boolean aliases are normalized.
- Candidate/selected-edge direct-provenance coverage: diagnostic set coverage, not
  evidence that the selected triple actually expresses the required fact.
- Reader trace stores the exact truncated context; document-level support does not
  guarantee the needed sentence survived truncation. Sentence-level support evaluation
  is not implemented, particularly where VN translation breaks sentence alignment.
- Query retrieval/answer latency, LLM token usage when supplied by server, fallback
  counts, context-drop counts, graph size and setup sessions.
- Complete arms only, identical QIDs only. Oracle/no-context are separate. Paired
  question bootstrap gives exploratory 95% intervals; no multiple-testing correction.

Splits are stratified by question type, **not support-document-disjoint**. Shared
support documents are counted in `splits.json`. A pooled retrieval corpus is intentional,
but overlapping questions/documents can make independent-question intervals optimistic.
For a final research claim, consider grouped splits/bootstrap and additional seeds.

Manual audit: use `audit_template.json`, two reviewers where feasible, mark supported/
unsupported/uncertain and error type. It includes stratum population counts; report
per-stratum precision or population-weighted estimates, not an unweighted overall
precision from equal-size strata. Missing knowledge requires a separate source-to-graph
coverage audit; checking extracted triples alone measures precision, not recall.

## Next steps (not implemented by this setup)

1. Complete the small QA smoke test and audit evidence before spending on a larger KG.
2. Freeze a representative dev/test dataset and run baseline matrix.
3. Select candidate quota/weight/cap on dev; evaluate only frozen choices on test.
4. If a gain persists, add contextual concept canonicalization or selective extraction
   verification with matched token budgets, multiple construction seeds and cost accounting.
5. Audit Vietnamese translations/gold answers; never silently repair labels using predictions.

This setup alone is not a novel-method claim; related-work review and larger controlled
experiments remain necessary.
