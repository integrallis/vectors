#!/usr/bin/env python3

import copy
import hashlib
import json
from pathlib import Path
import sys


if len(sys.argv) != 3:
    raise SystemExit(
        "usage: update_notebook_outputs.py <notebook-dir> <executed-dir>"
    )

notebook_dir = Path(sys.argv[1])
executed_dir = Path(sys.argv[2])
notebook_files = sorted(notebook_dir.glob("[0-9][0-9]_*.ipynb"))

if not notebook_files:
    raise RuntimeError(f"No notebooks found in {notebook_dir}")


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


for notebook_file in notebook_files:
    executed_file = executed_dir / notebook_file.name
    if not executed_file.exists():
        raise RuntimeError(f"Executed notebook is missing: {executed_file}")

    with notebook_file.open(encoding="utf-8") as stream:
        notebook = json.load(stream)
    with executed_file.open(encoding="utf-8") as stream:
        executed = json.load(stream)

    cells = notebook.get("cells", [])
    executed_cells = executed.get("cells", [])
    if len(cells) != len(executed_cells):
        raise RuntimeError(
            f"{notebook_file.name} has {len(cells)} source cells but "
            f"{len(executed_cells)} executed cells"
        )

    code_cell_count = 0
    source_digest = hashlib.sha256()
    for index, (cell, executed_cell) in enumerate(
        zip(cells, executed_cells, strict=True), start=1
    ):
        if cell.get("cell_type") != executed_cell.get("cell_type"):
            raise RuntimeError(
                f"{notebook_file.name} cell {index} changed type during execution"
            )
        if source_text(cell) != source_text(executed_cell):
            raise RuntimeError(
                f"{notebook_file.name} cell {index} source differs from the "
                "executed notebook; execute the current source again"
            )

        source_digest.update(str(cell.get("cell_type", "")).encode())
        source_digest.update(b"\0")
        source_digest.update(source_text(cell).encode())
        source_digest.update(b"\0")
        cell.get("metadata", {}).pop("execution", None)
        if cell.get("cell_type") == "code":
            code_cell_count += 1
            cell["execution_count"] = executed_cell.get("execution_count")
            cell["outputs"] = copy.deepcopy(executed_cell.get("outputs", []))

    vectors_metadata = notebook.setdefault("metadata", {}).setdefault("vectors", {})
    vectors_metadata["execution_source_sha256"] = source_digest.hexdigest()

    with notebook_file.open("w", encoding="utf-8") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")

    print(
        f"Updated {notebook_file.name}: "
        f"{code_cell_count} executed code cell(s)"
    )
