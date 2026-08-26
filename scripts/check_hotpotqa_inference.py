"""Static validation for HotpotQA inference-from-ZIP files."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_hotpotqa_from_zip.py"
NOTEBOOK = ROOT / "colab/AutoSchemaKG_HotpotQA_Inference_From_Zip.ipynb"
GUIDE = ROOT / "HOTPOTQA_INFERENCE.md"


def main() -> None:
    for path in (SCRIPT, NOTEBOOK, GUIDE):
        if not path.is_file():
            raise FileNotFoundError(path)
    ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Invalid or empty inference notebook")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
        "run_hotpotqa_from_zip.py",
        "Qwen/Qwen3.5-2B",
        "hotpotqa_kg_qa_results.json",
        "files.download",
    ):
        if required not in source:
            raise ValueError(f"Notebook is missing {required!r}")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    print("HotpotQA inference static validation passed.")


if __name__ == "__main__":
    main()
