#!/usr/bin/env python3

import json
import math
from pathlib import Path
import re
import sys


output_dir = Path(sys.argv[1])
expected_output = {
    "01_getting_started.ipynb": ["collection size: 500", "doc-374"],
    "02_quantization_tour.ipynb": [
        "NONE",
        "SQ8",
        "PQ",
        "BQ",
        "recall regression floors satisfied",
    ],
    "03_spring_ai_integration.ipynb": [
        "SIMD via Panama Vector API accelerates distance kernels on the JVM.",
        "after delete, collection size: 4",
    ],
    "04_langchain4j_integration.ipynb": [
        "[1] score=",
        "SIMD via Panama Vector API accelerates distance kernels on the JVM.",
    ],
    "05_embedding_cache.ipynb": [
        "after phase 1",
        "delegateCalls=4",
        "after phase 2",
        "delegateCalls=5",
        "after phase 3",
    ],
    "06_vcr_test_harness.ipynb": [
        "recorded under key: vcr:embedding:notebook-06:0000",
        "model=text-embedding-3-small",
        "vec[0]=0.110000",
        "store.listByTestId('notebook-06')",
        "temporary cassette directory removed",
    ],
}


def text_fragments(value):
    if isinstance(value, list):
        return [str(fragment) for fragment in value]
    if value is None:
        return []
    return [str(value)]


for name in expected_output:
    file = output_dir / name
    if not file.exists():
        raise RuntimeError(f"Executed notebook is missing: {file}")

    with file.open(encoding="utf-8") as stream:
        notebook = json.load(stream)

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if not code_cells:
        raise RuntimeError(f"{name} has no code cells")

    output_text = []
    for index, cell in enumerate(code_cells, start=1):
        if not isinstance(cell.get("execution_count"), int):
            raise RuntimeError(f"{name} code cell {index} was not executed")

        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "error":
                raise RuntimeError(
                    f"{name} code cell {index}: "
                    f"{output.get('ename')}: {output.get('evalue')}"
                )
            if output_type == "stream":
                fragments = text_fragments(output.get("text"))
                if output.get("name") == "stderr":
                    raise RuntimeError(
                        f"{name} code cell {index} wrote to stderr: {''.join(fragments)}"
                    )
                output_text.extend(fragments)
            if output_type in {"execute_result", "display_data"}:
                output_text.extend(
                    text_fragments(output.get("data", {}).get("text/plain"))
                )

    combined = "".join(output_text)
    if re.search(r"\b(?:AssertionError|Exception|ERROR)\b", combined):
        raise RuntimeError(f"{name} contains suspicious exception/error output")
    if re.search(r"java\.io\.PrintStream@[0-9a-f]+", combined):
        raise RuntimeError(f"{name} contains an unintended PrintStream display value")

    for expected in expected_output[name]:
        if expected not in combined:
            raise RuntimeError(f"{name} is missing expected output: {expected}")

    if name == "02_quantization_tour.ipynb":
        floors = {"NONE": 0.99, "SQ8": 0.95, "PQ": 0.95, "BQ": 0.55}
        for quantizer, floor in floors.items():
            match = re.search(
                rf"{quantizer}\s*-> recall@10 =\s*([0-9.]+)", combined
            )
            recall = float(match.group(1)) if match else math.nan
            if not math.isfinite(recall) or recall < floor or recall > 1.0:
                raise RuntimeError(
                    f"{name} has invalid {quantizer} recall: {recall}"
                )

    print(
        f"Validated {name}: {len(code_cells)} code cells, "
        f"{len(output_text)} output fragment(s)"
    )
