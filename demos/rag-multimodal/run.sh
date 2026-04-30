#!/usr/bin/env bash
#
# run.sh — Start vectors-server via Docker Compose, wait for health, launch the JavaFX demo.
#
# Usage:
#   cd vectors/demos/rag-multimodal
#   ./run.sh
#
# Ctrl+C stops the demo and offers to tear down the Docker container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VECTORS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HEALTH_URL="http://localhost:8287/v1/health"
TIMEOUT_SECS=120
POLL_INTERVAL=3

# ── Helpers ──────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✔ %s\033[0m\n' "$*"; }
err()   { printf '\033[1;31m✖ %s\033[0m\n' "$*" >&2; }

cleanup() {
    local exit_code=$?
    echo ""
    info "Demo stopped."
    read -r -p "Stop vectors-server container? [Y/n] " answer </dev/tty
    case "${answer:-Y}" in
        [Nn]*) info "Container left running. Stop later with: docker compose down" ;;
        *)     info "Stopping containers..."
               docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
               ok "Containers stopped." ;;
    esac
    exit "$exit_code"
}

# ── Main ─────────────────────────────────────────────────────────────────

# 1. Start Docker Compose (builds image on first run)
info "Starting vectors-server via Docker Compose..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d --build

# 2. Poll health endpoint
info "Waiting for vectors-server to become healthy (timeout: ${TIMEOUT_SECS}s)..."
elapsed=0
while true; do
    if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
        ok "vectors-server is healthy."
        break
    fi

    elapsed=$((elapsed + POLL_INTERVAL))
    if [ "$elapsed" -ge "$TIMEOUT_SECS" ]; then
        err "Timed out after ${TIMEOUT_SECS}s waiting for $HEALTH_URL"
        echo ""
        err "Last 40 lines of server logs:"
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" logs --tail=40 vectors-server
        exit 1
    fi

    printf "  … %ds / %ds\r" "$elapsed" "$TIMEOUT_SECS"
    sleep "$POLL_INTERVAL"
done

# 3. Launch the JavaFX demo on the host
trap cleanup INT TERM
info "Launching rag-multimodal demo..."
"$VECTORS_ROOT/gradlew" -p "$VECTORS_ROOT" :demos:rag-multimodal:run
