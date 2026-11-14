#!/usr/bin/env bash
# Confirm run: v0.1.34, v0.1.35, fixA (static nested scorers, final sim/vectors), fixB (A + final
# query scratch copy) on AnnSearchBenchmark.searchHnsw and IndexBuildBenchmark.buildHnsw; 2 rounds,
# arms interleaved per benchmark; fork=2 warmup=3 iterations=5. fixA/fixB are v0.1.35 worktrees
# with the patch applied (patches kept beside the results).
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/confirm; mkdir -p $OUT; cp ~/fixA.patch ~/fixB.patch $OUT/
cd ~/vectors
for fx in fixA fixB; do
  [ -d ~/p/$fx ] || { git worktree add -q --detach ~/p/$fx v0.1.35 && (cd ~/p/$fx && git apply ~/$fx.patch && echo "$fx applied: $(git diff --stat | tail -1)"); }
  (cd ~/p/$fx && ./gradlew :vectors-bench:compileJmhJava --no-daemon -q > ~/p/$fx.build.log 2>&1; echo "built $fx exit=$?")
done
ARMS=(v0.1.34 v0.1.35 fixA fixB)
BENCH=("AnnSearchBenchmark.searchHnsw" "IndexBuildBenchmark.buildHnsw")
echo "=== confirm: ${#BENCH[@]} benchmarks x ${#ARMS[@]} arms x 2 rounds  $(date -u +%T)"
for r in 1 2; do for b in "${BENCH[@]}"; do
  re="${b%.*}\\.${b##*.}\$"; stem=${b//./_}
  for a in "${ARMS[@]}"; do
    name=$a-$stem-r$r
    (cd ~/p/$a && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes="$re" -Pjmh.resultFormat=JSON -Pjmh.fork=2 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/$name.json; echo "  $name exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
  done
done; done
echo "CONFIRM-DONE $(date -u +%FT%TZ)"
