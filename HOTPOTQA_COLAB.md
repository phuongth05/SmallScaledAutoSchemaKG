# HotpotQA on personal Colab

Open `colab/AutoSchemaKG_HotpotQA_Qwen35_2B.ipynb` to build a small
AutoSchemaKG graph from the official `hotpotqa/hotpot_qa` dataset with a local
`Qwen/Qwen3.5-2B` server.

## What this experiment does

1. Streams a deterministic contiguous slice of HotpotQA from Hugging Face.
2. Writes Wikipedia context passages to `hotpotqa_corpus.json`.
3. Writes questions, answers, and gold supporting facts separately to
   `qa_manifest.json`; answers are not included in the KG construction input.
4. Extracts entity and event triples, induces concepts, and exports GraphML.
5. Packages the graph outputs and dataset provenance into
   `autoschemakg_hotpotqa_v1.zip`.

The default experiment uses `distractor/validation`, 3 questions, and all ten
context passages per question. Set `CONTEXT_MODE = "supporting"` for a much
faster oracle-corpus smoke test. That mode is useful for debugging but must not
be reported as a realistic distractor evaluation.

## Colab sizing

Start with these settings on a T4-class runtime:

| Goal | `MAX_QUESTIONS` | `CONTEXT_MODE` | Expected corpus |
| --- | ---: | --- | --- |
| Pipeline check | 1 | `supporting` | about 2 documents |
| First experiment | 3 | `all` | up to 30 documents |
| Larger trial | 10 | `all` | up to 100 documents |

Every document can trigger three extraction generations plus concept
generation. Runtime therefore grows much faster than the number of questions.
The complete 90,447-question training split is not a personal-Colab target.

## Reproducibility artifacts

The result archive contains:

- `run_summary.json`: model, graph counts, and embedded dataset metadata;
- `kg_extraction/`, `triples_csv/`, `concepts/`, `concept_csv/`, `kg_graphml/`;
- `provenance/hotpotqa_corpus.json`: exact corpus slice passed to extraction;
- `provenance/qa_manifest.json`: questions, answers, and supporting facts;
- `provenance/dataset_metadata.json`: config, split, offset, counts, and license.

This proves that a specific HotpotQA slice completed the KG construction
pipeline. It does not by itself measure HotpotQA answer accuracy. QA retrieval,
answer generation, and EM/F1 scoring are a later stage.

## Command-line preparation

```bash
python scripts/prepare_hotpotqa.py \
  --config distractor \
  --split validation \
  --max-questions 3 \
  --context-mode all \
  --output-dir data/hotpotqa_v1 \
  --overwrite
```

HotpotQA is distributed under CC BY-SA 4.0. Preserve dataset attribution when
sharing prepared corpus artifacts.
