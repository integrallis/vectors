#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Benchmark comparison runner: BASELINE (default: previous release tag) vs CANDIDATE (default: main)
# ---------------------------------------------------------------------------
# Run on a quiet VPS (Hetzner or equivalent) for stable measurements.
#
# Usage:
#   cd /path/to/vectors
#   BASELINE=v0.1.35 CANDIDATE=main ./experiments/benchmark-comparison.sh [output_dir]
#   (ROUNDS, JMH_FORK, JMH_WARMUP, JMH_ITERATIONS override the defaults 2 / 5 / 3 / 5)
#
# The script:
#  1. Records machine specs
#  2. Adds a detached worktree for each ref and builds the JMH classes once
#  3. Runs the benchmark list, interleaving baseline and candidate per benchmark and round
#  4. Collects JMH JSON outputs into output_dir/<ref>/
# ---------------------------------------------------------------------------

set -euo pipefail

BASELINE="${BASELINE:-$(git describe --tags --abbrev=0 main)}"
CANDIDATE="${CANDIDATE:-main}"
OUTPUT_DIR="${1:-benchmark-results/$(date +%Y-%m-%d)-perf-comparison}"
mkdir -p "$OUTPUT_DIR"

echo "=== Benchmark comparison: $BASELINE vs $CANDIDATE ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Record machine specs
{
  echo "# Benchmark comparison"
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Runner: $(whoami)@$(hostname)"
  echo "Java: $(java -version 2>&1 | head -1)"
  echo "OS: $(uname -a)"
  echo "CPUs: $(nproc)"
  echo "Memory: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'unknown')"
  echo "Baseline $BASELINE: $(git rev-parse "$BASELINE")"
  echo "Candidate $CANDIDATE: $(git rev-parse "$CANDIDATE")"
  echo ""
} > "$OUTPUT_DIR/META.txt"

cat "$OUTPUT_DIR/META.txt"

# JMH benchmarks to run. Each entry is Class.method; a trailing "." selects every method of the
# class. JMH matches the include regex with find() against the fully-qualified benchmark name, so
# the pattern must not be anchored at the start (an earlier version of this script used
# "^${bench}$", which never matched anything and produced no results).
BENCHMARKS=(
  "AnnSearchBenchmark.searchFlat"
  "AnnSearchBenchmark.searchHnsw"
  "AnnSearchBenchmark.searchVamana"
  "AnnSearchBenchmark.searchIvfFlat"
  "IndexBuildBenchmark.buildHnsw"
  "IndexBuildBenchmark.buildVamana"
  "IndexBuildBenchmark.buildIvfFlat"
  "CommitPipelineBenchmark.copyOfAndStage"
  "GgufBatchedMatmulBenchmark.batched"
)
ROUNDS="${ROUNDS:-2}"
# Five forks by default: on the 2026-11-14 audit host the fork-to-fork spread of a 70 us HNSW search was
# +-6% on the same build, so two forks cannot resolve a 2-5% question (benchmark-results/2026-11-14-0.1.35-release-audit).
JMH_ARGS=(-Pjmh.resultFormat=JSON -Pjmh.fork="${JMH_FORK:-5}" -Pjmh.warmup="${JMH_WARMUP:-3}" -Pjmh.iterations="${JMH_ITERATIONS:-5}")

include_regex() {
  local bench="$1"
  if [[ "$bench" == *. ]]; then printf '%s\\.' "${bench%.}"; else printf '%s\\.%s$' "${bench%.*}" "${bench##*.}"; fi
}

# Both refs are checked out as detached worktrees so the invoking checkout is never touched
# (no stash, no checkout) and so the two arms can be interleaved per benchmark.
WORKTREES="$(mktemp -d)"
trap 'for ref in "$BASELINE" "$CANDIDATE"; do git worktree remove --force "$WORKTREES/$ref" 2>/dev/null || true; done; rmdir "$WORKTREES" 2>/dev/null || true' EXIT
for ref in "$BASELINE" "$CANDIDATE"; do
  git worktree add --detach "$WORKTREES/$ref" "$ref"
  echo "=== [$ref] Building ==="
  (cd "$WORKTREES/$ref" && ./gradlew :vectors-bench:compileJmhJava --no-daemon -q)
done

# Interleave: for every round and benchmark, baseline then candidate, so drift on the host hits
# both arms alike. The build writes vectors-bench/build/results/jmh/results.json (see
# vectors-bench/build.gradle.kts); it is copied out under <ref>/<bench>-r<round>.json.
for round in $(seq 1 "$ROUNDS"); do
  for bench in "${BENCHMARKS[@]}"; do
    for ref in "$BASELINE" "$CANDIDATE"; do
      out_dir="$OUTPUT_DIR/$ref"; mkdir -p "$out_dir"
      stem="${bench%.}"; stem="${stem//./_}"
      echo ""
      echo "=== [$ref] round $round: $bench ==="
      (
        cd "$WORKTREES/$ref"
        rm -rf vectors-bench/build/results/jmh
        ./gradlew :vectors-bench:jmh -Pjmh.includes="$(include_regex "$bench")" "${JMH_ARGS[@]}" --no-daemon \
          2>&1 | tee "$out_dir/$stem-r$round.log"
        cp vectors-bench/build/results/jmh/results.json "$out_dir/$stem-r$round.json"
      ) || echo "!!! [$ref] $bench round $round FAILED (see $out_dir/$stem-r$round.log)"
    done
  done
done

echo ""
echo "=== Results collected in $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR/"
