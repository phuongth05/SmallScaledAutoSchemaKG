"""Static validation for files required by the Colab v1 workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "requirements-colab.txt",
    "scripts/run_colab_v1.py",
    "example/example_data/v1_smoke.json",
    "colab/AutoSchemaKG_v1_Qwen35_2B.ipynb",
    "COLAB_V1.md",
    "REPRODUCTION.md",
)


def main() -> None:
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)

    ast.parse((ROOT / "scripts/run_colab_v1.py").read_text(encoding="utf-8"))
    data = json.loads(
        (ROOT / "example/example_data/v1_smoke.json").read_text(encoding="utf-8")
    )
    if not data or not data[0].get("text"):
        raise ValueError("Smoke dataset is empty")
    notebook = json.loads(
        (ROOT / "colab/AutoSchemaKG_v1_Qwen35_2B.ipynb").read_text(encoding="utf-8")
    )
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Invalid or empty notebook")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    print("Colab v1 static validation passed.")


if __name__ == "__main__":
    main()
