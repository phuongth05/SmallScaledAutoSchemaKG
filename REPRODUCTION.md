# Small-scale reproduction plan

The repository is organized around four incremental stages. Each stage keeps
the same relative-path and local-endpoint conventions used by Colab v1.

## Stage 1: construction smoke test (implemented)

- Input: one short document in `example/example_data/v1_smoke.json`.
- Local constructor: `Qwen/Qwen3.5-2B` served by vLLM.
- Artifacts: extraction JSON, triple/concept CSV files, GraphML, run summary.
- Entry point: `scripts/run_colab_v1.py`.

Acceptance criteria:

- the local model endpoint responds;
- extraction JSON is produced;
- GraphML is readable by NetworkX;
- the run summary reports non-zero nodes and edges.

## Stage 2: construction ablation (prepared next)

Run the same fixed document sample as:

- entity only;
- entity + event;
- entity + event + concept.

Record model revision, seed, elapsed time, node/edge counts, and malformed JSON
rate. Stage 1 outputs establish the data contract needed for these runs.

## Stage 3: small multi-hop QA (prepared next)

- Use 50-200 fixed questions from MuSiQue, HotpotQA, and 2WikiMultiHopQA.
- Compare no retrieval, dense retrieval, and HippoRAG2.
- Reuse the GraphML/CSV layout produced in Stage 1.
- Keep the question sample and seed in version control; keep embeddings outside
  Git when they become large.

## Stage 4: schema quality (prepared next)

- Start with FB15kET before YAGO43kET and wikiHow.
- Freeze the prompt, model revision, decoding configuration, and sampling seed.
- Report BS-R and BS-C together with failure counts.

Full ATLAS reconstruction and hosting are explicitly out of scope for personal
Colab/Kaggle runtimes because the released artifacts and compute requirements
are at terabyte and multi-GPU scale.
