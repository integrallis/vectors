#!/usr/bin/env bash
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
export VECTORS_RESEARCH_DIR=$HOME/research
OUT=$HOME/out; mkdir -p $OUT/jmh $OUT/jmh-failed-attempt
# Corrected JMH phase. The first attempt (copied from the repo's experiments/benchmark-comparison.sh)
# failed every run: `^Class.method$` never matches JMH's fully-qualified benchmark names, and
# -Pjmh.output / build/reports/jmh are not what vectors-bench/build.gradle.kts reads or writes
# (it reads -Pjmh.resultFormat and writes build/results/jmh/results.<ext>).
mv $OUT/jmh/*.log $OUT/jmh-failed-attempt/ 2>/dev/null
BENCH=("AnnSearchBenchmark.searchFlat" "AnnSearchBenchmark.searchHnsw" "AnnSearchBenchmark.searchVamana" "AnnSearchBenchmark.searchIvfFlat" "IndexBuildBenchmark.buildHnsw" "IndexBuildBenchmark.buildVamana" "IndexBuildBenchmark.buildIvfFlat" "CommitPipelineBenchmark.copyOfAndStage" "GgufBatchedMatmulBenchmark.batched" "GgufQ4KBatchedMatmulBenchmark.")
echo "=== JMH: ${#BENCH[@]} selectors x 2 tags x 2 rounds, interleaved per selector; fork=2 warmup=3 iterations=5  $(date -u +%T)"
for r in 1 2; do for b in "${BENCH[@]}"; do
  if [[ $b == *. ]]; then re="${b%.}\\."; else re="${b%.*}\\.${b##*.}\$"; fi
  stem=${b%.}
  for t in v0.1.34 v0.1.35; do
    name=$t-${stem//./_}-r$r
    (cd ~/$t && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes="$re" -Pjmh.resultFormat=JSON -Pjmh.fork=2 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/jmh/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/jmh/$name.json; echo "  $t $b r$r exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
  done
done; done
echo "VECJMH-DONE $(date -u +%FT%TZ)"
