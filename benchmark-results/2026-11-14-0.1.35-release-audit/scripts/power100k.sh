#!/usr/bin/env bash
# Powered A/B at the one shape still resolved by two forks only: VectorDbBenchmark.searchRawHnsw and
# searchFacadeHnsw at n=100000, dim=128, v0.1.34 vs v0.1.35, fork=5 warmup=3 iterations=5, 3 interleaved
# rounds (15 forks per arm). The @Param grids are narrowed identically in both tags' bench source.
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/power100k; mkdir -p $OUT
for t in v0.1.34 v0.1.35; do
  f=~/$t/vectors-bench/src/jmh/java/com/integrallis/vectors/bench/VectorDbBenchmark.java
  sed -i 's/@Param({"10000", "100000"})/@Param({"100000"})/; s/@Param({"128", "768"})/@Param({"128"})/' $f
  echo "$t params: n=$(grep -c '@Param({"100000"})' $f) dim=$(grep -c '@Param({"128"})' $f)"
  (cd ~/$t && ./gradlew :vectors-bench:compileJmhJava --no-daemon -q > ~/$t.power100k-build.log 2>&1; echo "built $t exit=$?")
done
echo "=== power100k: searchRawHnsw+searchFacadeHnsw 100k x 128, 2 tags x 3 rounds, fork=5  $(date -u +%T)"
for r in 1 2 3; do for t in v0.1.34 v0.1.35; do
  name=$t-VectorDbBenchmark-r$r
  (cd ~/$t && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes='VectorDbBenchmark\.' -Pjmh.resultFormat=JSON -Pjmh.fork=5 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/$name.json; echo "  $name exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
done; done
echo "POWER100K-DONE $(date -u +%FT%TZ)"
