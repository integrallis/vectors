#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Benchmark comparison runner: v0.1.34 (prior release) vs main (current HEAD)
# ---------------------------------------------------------------------------
# Run on a quiet VPS (Hetzner or equivalent) for stable measurements.
#
# Usage:
#   cd /path/to/vectors
#   ./experiments/benchmark-comparison.sh [output_dir]
#
# The script:
#  1. Records machine specs
#  2. Checks out v0.1.34, builds, runs the 5 targeted benchmarks
#  3. Checks out main, builds, runs the same 5 benchmarks
#  4. Collects JMH JSON outputs into output_dir
# ---------------------------------------------------------------------------

set -euo pipefail

OUTPUT_DIR="${1:-benchmark-results/$(date +%Y-%m-%d)-perf-comparison}"
mkdir -p "$OUTPUT_DIR"

echo "=== Benchmark comparison: v0.1.34 vs main ==="
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
  echo "Git v0.1.34: $(git rev-parse v0.1.34)"
  echo "Git main HEAD: $(git rev-parse main)"
  echo ""
} > "$OUTPUT_DIR/META.txt"

cat "$OUTPUT_DIR/META.txt"

# JMH benchmarks to run
BENCHMARKS=(
  "AnnSearchBenchmark.searchHnsw"
  "AnnSearchBenchmark.searchVamana"
  "IndexBuildBenchmark.buildHnsw"
  "IndexBuildBenchmark.buildVamana"
  "CommitPipelineBenchmark.copyOfAndStage"
)

run_benchmarks() {
  local tag="$1"
  local out_dir="$OUTPUT_DIR/$tag"
  mkdir -p "$out_dir"

  echo ""
  echo "=== [$tag] Building ==="
  ./gradlew :vectors-bench:compileJmhJava --no-daemon -q

  for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=== [$tag] Running $bench ==="
    ./gradlew :vectors-bench:jmh \
      -Pjmh.includes="^${bench}$" \
      -Pjmh.output=json \
      --no-daemon \
      2>&1 | tee "$out_dir/${bench//\//_}.log"
  done

  # Copy generated JMH JSON results
  find vectors-bench/build/reports/jmh -name "*.json" -exec cp {} "$out_dir/" \; || true
}

# Save current branch
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo ""
echo "=== Baseline: v0.1.34 ==="
git stash push -m "benchmark stash" --include-untracked || true
git checkout v0.1.34
run_benchmarks "v0.1.34"

echo ""
echo "=== Candidate: main ==="
git checkout main
run_benchmarks "main"

# Restore original branch
git checkout "$ORIGINAL_BRANCH"
git stash pop || true

echo ""
echo "=== Results collected in $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR/"
