#!/usr/bin/env bash
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
export VECTORS_RESEARCH_DIR=$HOME/research
OUT=$HOME/out; mkdir -p $OUT/jmh $OUT/recall $OUT/slow
echo "=== vectors 0.1.34 vs 0.1.35 audit  $(date -u +%FT%TZ)"
lscpu > $OUT/cpu.txt; java -version > $OUT/java-version.txt 2>&1
# siftsmall for the recall tests, at the path DatasetRegistry's built-in lookup uses.
mkdir -p ~/research/repos/jvector && cd ~/research/repos/jvector && curl -sL -o siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz && tar xzf siftsmall.tar.gz && ls siftsmall | tr '\n' ' '; echo
cd ~/vectors && for t in v0.1.34 v0.1.35; do git worktree add -q --detach ~/$t $t; (cd ~/$t && ./gradlew :vectors-bench:compileJmhJava :vectors-bench:testClasses --no-daemon -q 2>&1 | grep -viE "WARNING|restricted" | tail -2); echo "built $t $(git -C ~/$t rev-parse --short HEAD)"; done
BENCH=("AnnSearchBenchmark.searchHnsw" "AnnSearchBenchmark.searchVamana" "IndexBuildBenchmark.buildHnsw" "IndexBuildBenchmark.buildVamana" "CommitPipelineBenchmark.copyOfAndStage" "GgufBatchedMatmulBenchmark.batched" "GgufQ4KBatchedMatmulBenchmark.*")
echo "=== JMH: 7 benchmarks x 2 tags x 2 rounds, interleaved per benchmark; fork=2 warmup=3 iterations=5  $(date -u +%T)"
for r in 1 2; do for b in "${BENCH[@]}"; do for t in v0.1.34 v0.1.35; do
  (cd ~/$t && rm -rf vectors-bench/build/reports/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes="^${b}$" -Pjmh.output=json -Pjmh.fork=2 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/jmh/$t-${b//[.*]/_}-r$r.log 2>&1; rc=$?; f=$(find vectors-bench/build/reports/jmh -name "*.json" | head -1); [ -n "$f" ] && cp "$f" $OUT/jmh/$t-${b//[.*]/_}-r$r.json; echo "  $t $b r$r exit=$rc json=$([ -n "$f" ] && echo yes || echo NO) $(date -u +%T)")
done; done; done
echo "=== recallGate: verify on v0.1.35 against the committed k-means-era baseline; record on both  $(date -u +%T)"
(cd ~/v0.1.35 && ./gradlew :vectors-bench:recallGate --no-daemon > $OUT/recall/v0.1.35-verify.log 2>&1; echo "  v0.1.35 verify exit=$?"; cp vectors-bench/build/test-results/recallGate/*.xml $OUT/recall/v0.1.35-verify.xml 2>/dev/null)
for t in v0.1.34 v0.1.35; do (cd ~/$t && ./gradlew :vectors-bench:recallGate -Drecall.gate.mode=record --no-daemon > $OUT/recall/$t-record.log 2>&1; echo "  $t record exit=$?"); done
echo "=== SIFT-small HNSW/Vamana recall floors on both tags  $(date -u +%T)"
for t in v0.1.34 v0.1.35; do (cd ~/$t && ./gradlew :vectors-bench:slowTest --tests '*SiftSmall*RecallTest' --no-daemon > $OUT/slow/$t.log 2>&1; echo "  $t slow exit=$?"; mkdir -p $OUT/slow/$t && cp vectors-bench/build/test-results/slowTest/*.xml $OUT/slow/$t/ 2>/dev/null); done
echo "VECDRIVER-DONE $(date -u +%FT%TZ)"
