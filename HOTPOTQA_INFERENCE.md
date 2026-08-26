# HotpotQA inference from an AutoSchemaKG ZIP

Use `colab/AutoSchemaKG_HotpotQA_Inference_From_Zip.ipynb` to inspect a saved
HotpotQA graph and run three-question KG-only QA without rebuilding the graph.

## Required ZIP contents

The input is the archive produced by the construction notebook. It must contain:

- `kg_graphml/hotpotqa_corpus_graph.graphml`;
- `provenance/hotpotqa_corpus.json`;
- `provenance/qa_manifest.json`;
- `run_summary.json` (recommended).

Do not upload a ZIP of the entire Colab `/content` directory. Such an archive
contains caches, logs, a duplicate repository (including `.git`), and nested
archives. It is unnecessary and may expose runtime configuration.

## Workflow

1. Select a GPU runtime before uploading the ZIP.
2. Upload the construction-result ZIP.
3. Clone/install the repository and start local `Qwen/Qwen3.5-2B` through vLLM.
4. Safely extract the ZIP and validate required artifacts.
5. Print graph statistics and save an 80-node overview PNG.
6. Retrieve up to 60 graph triples per question from its ten distractor-context
   documents.
7. Generate short answers with Qwen, calculate Exact Match and token F1, and
   write `hotpotqa_kg_qa_results.json`.
8. Download the reanalyzed ZIP.

Gold answers and supporting-fact labels are used only for post-generation
evaluation. They are not included in the LLM prompt.

## Important ID mapping

AutoSchemaKG hashes passage IDs while converting extraction JSON to CSV and
GraphML. The provenance IDs such as `hotpotqa-c1b...` therefore do not equal
GraphML `file_id` values. The inference script maps each provenance document to
its passage node through exact corpus text, then retrieves triples using the
hashed GraphML passage ID.

An earlier ad-hoc notebook compared the two ID namespaces directly. It returned
zero triples for every question, so its EM/F1 result is invalid: the model
answered without graph context. `scripts/run_hotpotqa_from_zip.py` raises an
error when retrieval is empty instead of silently evaluating such a run.

The inspected construction artifact itself is valid and contains 6,330 nodes
(430 entities, 322 events, 30 passages, and 5,548 concepts) plus 19,189 edges.
After correcting the ID mapping, the three questions expose 300, 250, and 252
candidate relation triples respectively; the script ranks and sends the top 60.

## CLI

With a local OpenAI-compatible server already listening on port 8000:

```bash
python -u scripts/run_hotpotqa_from_zip.py autoschemakg_hotpotqa_v1.zip \
  --work-dir outputs/hotpotqa_from_zip \
  --output-zip outputs/autoschemakg_hotpotqa_reanalyzed.zip \
  --model Qwen/Qwen3.5-2B \
  --base-url http://127.0.0.1:8000/v1 \
  --overwrite
```

Graph inspection alone does not require a GPU or model server:

```bash
python scripts/run_hotpotqa_from_zip.py autoschemakg_hotpotqa_v1.zip \
  --work-dir outputs/hotpotqa_visualization \
  --visualize-only \
  --overwrite
```

## Interpretation

The first experiment contains only three hard validation questions and thirty
Wikipedia context passages. It is a pipeline validation, not a statistically
meaningful HotpotQA result. Report graph counts, retrieved-triple counts, EM/F1,
model checkpoint, and exact dataset slice together.
