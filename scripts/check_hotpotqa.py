"""Static checks for the HotpotQA Colab workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "scripts/prepare_hotpotqa.py",
    "scripts/run_colab_v1.py",
    "colab/AutoSchemaKG_HotpotQA_Qwen35_2B.ipynb",
    "HOTPOTQA_COLAB.md",
)


def main() -> None:
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)

    for relative in ("scripts/prepare_hotpotqa.py", "scripts/run_colab_v1.py"):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)

    notebook_path = ROOT / "colab/AutoSchemaKG_HotpotQA_Qwen35_2B.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise ValueError("Invalid or empty HotpotQA notebook")
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for required_text in (
        "scripts/prepare_hotpotqa.py",
        "hotpotqa_corpus",
        "dataset_metadata.json",
        "scripts/run_colab_v1.py",
        "autoschemakg_hotpotqa_v1",
    ):
        if required_text not in source:
            raise ValueError(f"Notebook is missing {required_text!r}")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    print("HotpotQA Colab static validation passed.")


if __name__ == "__main__":
    main()
