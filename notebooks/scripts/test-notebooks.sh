#!/usr/bin/env bash

set -euo pipefail

repository=${VECTORS_REPOSITORY:-/home/jovyan/work/vectors}
notebook_dir="$repository/notebooks"
output_dir=${VECTORS_NOTEBOOK_OUTPUT_DIR:-"$repository/build/notebooks/executed"}
mode=${VECTORS_NOTEBOOK_MODE:-source}

case "$mode" in
    source)
        notebooks=("$notebook_dir"/*.ipynb)
        ;;
    release)
        notebooks=(
            "$notebook_dir"/01_getting_started.ipynb
            "$notebook_dir"/02_quantization_tour.ipynb
            "$notebook_dir"/03_spring_ai_integration.ipynb
            "$notebook_dir"/04_langchain4j_integration.ipynb
            "$notebook_dir"/05_embedding_cache.ipynb
            "$notebook_dir"/06_vcr_test_harness.ipynb
        )
        ;;
    *)
        printf 'Unsupported VECTORS_NOTEBOOK_MODE: %s (expected source or release)\n' "$mode" >&2
        exit 2
        ;;
esac

mkdir -p "$output_dir"

for notebook in "${notebooks[@]}"; do
    name=$(basename "$notebook")
    printf 'Executing %s (%s dependencies)\n' "$name" "$mode"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.kernel_name=java \
        --ExecutePreprocessor.timeout=600 \
        --output-dir="$output_dir" \
        --output="$name" \
        "$notebook"
done

python "$notebook_dir/scripts/validate_notebook_outputs.py" "$output_dir"

printf 'Executed %d notebook(s); outputs: %s\n' "${#notebooks[@]}" "$output_dir"
