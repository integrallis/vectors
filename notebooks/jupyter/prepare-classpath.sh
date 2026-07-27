#!/usr/bin/env bash

if [[ "${VECTORS_NOTEBOOK_PREPARE:-true}" != "true" ]]; then
    return 0
fi

repository=/home/jovyan/work/vectors
mode=${VECTORS_NOTEBOOK_MODE:-source}
version=${VECTORS_VERSION:-0.1.0}
repository_url=${VECTORS_NOTEBOOK_REPOSITORY:-}

case "$mode" in
    source)
        if ! "$repository/gradlew" \
            --no-daemon \
            -p "$repository" \
            prepareNotebookClasspath \
            -PnotebookMode=source; then
            return 1
        fi
        ;;
    release)
        args=(
            --no-daemon
            -p "$repository"
            prepareNotebookClasspath
            -PnotebookMode=release
            "-PnotebookVersion=$version"
        )
        if [[ -n "$repository_url" ]]; then
            args+=("-PnotebookRepository=$repository_url")
        fi
        if ! "$repository/gradlew" "${args[@]}"; then
            return 1
        fi
        ;;
    *)
        printf 'Unsupported VECTORS_NOTEBOOK_MODE: %s (expected source or release)\n' "$mode" >&2
        return 2
        ;;
esac

unset repository mode version repository_url args
