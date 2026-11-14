#!/usr/bin/env bash
# Power run: the staircase showed point-to-point swings of +-4% with 4 forks per point, i.e. a 2-3%
# endpoint difference is below what fork=2 x 2 rounds resolves. Here: v0.1.34, v0.1.35, fixA on
# AnnSearchBenchmark.searchHnsw with n in {2000, 50000} (the benchmark's @Param widened identically in
# every arm's bench source; library code untouched), fork=5 warmup=3 iterations=5, 3 interleaved rounds
# -> 15 forks per arm per shape.
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/power; mkdir -p $OUT
ARMS=(v0.1.34 v0.1.35 fixA)
for a in "${ARMS[@]}"; do
  f=~/p/$a/vectors-bench/src/jmh/java/com/integrallis/vectors/bench/AnnSearchBenchmark.java
  grep -q '@Param({"2000", "50000"})' $f || sed -i 's/@Param({"2000"})/@Param({"2000", "50000"})/' $f
  grep -n -A1 '@Param({"2000", "50000"})' $f | head -2 | tr '\n' ' '; echo " <- $a"
  (cd ~/p/$a && ./gradlew :vectors-bench:compileJmhJava --no-daemon -q > ~/p/$a.power-build.log 2>&1; echo "built $a exit=$?")
done
echo "=== power: searchHnsw n={2000,50000} x 3 arms x 3 rounds, fork=5  $(date -u +%T)"
for r in 1 2 3; do for a in "${ARMS[@]}"; do
  name=$a-searchHnsw-r$r
  (cd ~/p/$a && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes='AnnSearchBenchmark\.searchHnsw$' -Pjmh.resultFormat=JSON -Pjmh.fork=5 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/$name.json; echo "  $name exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
done; done
echo "POWER-DONE $(date -u +%FT%TZ)"
