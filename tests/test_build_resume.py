from __future__ import annotations

import csv
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate_response(self, inputs, return_text_only=True, **kwargs):
        self.calls += len(inputs)
        values = ["concept"] * len(inputs)
        return values if return_text_only else [(value, {}) for value in values]


def load_concept_module(monkeypatch):
    llm = types.ModuleType("atlas_rag.llm_generator")
    llm.LLMGenerator = object
    config = types.ModuleType("atlas_rag.kg_construction.triple_config")
    config.ProcessingConfig = object
    graphml = types.ModuleType("atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml")
    graphml.get_node_id = lambda value: value
    prompt = types.ModuleType("atlas_rag.llm_generator.prompt.triple_extraction_prompt")
    prompt.CONCEPT_INSTRUCTIONS = {
        "en": {
            "event": "event [EVENT]", "entity": "entity [ENTITY] [CONTEXT]",
            "relation": "relation [RELATION]",
        }
    }
    for name, module in {
        "atlas_rag": types.ModuleType("atlas_rag"),
        "atlas_rag.llm_generator": llm,
        "atlas_rag.kg_construction": types.ModuleType("atlas_rag.kg_construction"),
        "atlas_rag.kg_construction.triple_config": config,
        "atlas_rag.kg_construction.utils": types.ModuleType("atlas_rag.kg_construction.utils"),
        "atlas_rag.kg_construction.utils.csv_processing": types.ModuleType("atlas_rag.kg_construction.utils.csv_processing"),
        "atlas_rag.kg_construction.utils.csv_processing.csv_to_graphml": graphml,
        "atlas_rag.llm_generator.prompt": types.ModuleType("atlas_rag.llm_generator.prompt"),
        "atlas_rag.llm_generator.prompt.triple_extraction_prompt": prompt,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "concept_generation_under_test", ROOT / "atlas_rag/kg_construction/concept_generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concept_generation_resumes_without_duplicate_rows(tmp_path, monkeypatch):
    generate_concept = load_concept_module(monkeypatch).generate_concept
    pattern = "corpus"
    triples = tmp_path / "triples_csv"
    concepts = tmp_path / "concepts"
    graph_dir = tmp_path / "kg_graphml"
    triples.mkdir()
    graph_dir.mkdir()
    source = triples / f"missing_concepts_{pattern}_from_json.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type"])
        writer.writerows([["event a", "Event"], ["event b", "Event"], ["relation a", "Relation"]])
    with (graph_dir / f"{pattern}_without_concept.pkl").open("wb") as stream:
        pickle.dump(nx.DiGraph(), stream)
    config = types.SimpleNamespace(
        output_directory=str(tmp_path), filename_pattern=pattern, max_workers=1,
    )

    concepts.mkdir()
    with (concepts / "concept_shard_0.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node", "conceptualized_node", "node_type"])
        writer.writerow(["event a", "saved concept", "event"])
    model = FakeModel()
    generate_concept(
        model, input_file=str(source), output_folder=str(concepts),
        output_file="concept.json", logging_file=str(concepts / "logging.txt"),
        config=config, batch_size=1,
    )
    assert model.calls == 2
    with (concepts / "concept_shard_0.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 4
    assert len({(row[0], row[2]) for row in rows[1:]}) == 3
