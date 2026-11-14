#!/usr/bin/env bash
# Revert run: each commit on the plain (non-multistart, full-precision) HNSW search path reverted
# individually on top of v0.1.35, measured on VectorDbBenchmark.searchRawHnsw at n=10000 dim=128 (the
# @Param grids narrowed identically in every arm's bench source), fork=5 warmup=3 iterations=5, 3
# interleaved rounds -> 15 forks per arm. Arms: v0.1.34, v0.1.35, fixA, and v0.1.35 minus one commit each.
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/revert; mkdir -p $OUT
cd ~/vectors
REVS=(871dafcc 73bcc1b8 97579da4 b500e86e)
ARMS=(v0.1.34 v0.1.35 fixA)
for c in "${REVS[@]}"; do
  a=rev-$c
  if [ ! -d ~/p/$a ]; then
    git worktree add -q --detach ~/p/$a v0.1.35
    if (cd ~/p/$a && git revert --no-commit $c > ~/p/$a.revert.log 2>&1); then echo "$a: reverted cleanly ($(cd ~/p/$a && git diff --cached --stat | tail -1))"; ARMS+=($a); else echo "$a: REVERT CONFLICT, arm skipped"; cat ~/p/$a.revert.log | tail -3; fi
  else ARMS+=($a); fi
done
for a in "${ARMS[@]}"; do
  f=~/p/$a/vectors-bench/src/jmh/java/com/integrallis/vectors/bench/VectorDbBenchmark.java
  sed -i 's/@Param({"10000", "100000"})/@Param({"10000"})/; s/@Param({"128", "768"})/@Param({"128"})/' $f
  echo "$a params: $(grep -c '@Param({"10000"})' $f)+$(grep -c '@Param({"128"})' $f)"
done
printf '%s\n' "${ARMS[@]}" | xargs -P 4 -I{} bash -c 'cd ~/p/{} && ./gradlew :vectors-bench:compileJmhJava --no-daemon -q > ~/p/{}.revert-build.log 2>&1; echo "built {} exit=$?"'
echo "=== revert run: ${#ARMS[@]} arms (${ARMS[*]}) x 3 rounds, fork=5, searchRawHnsw 10k x 128  $(date -u +%T)"
for r in 1 2 3; do for a in "${ARMS[@]}"; do
  name=$a-searchRawHnsw-r$r
  (cd ~/p/$a && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes='VectorDbBenchmark\.searchRawHnsw$' -Pjmh.resultFormat=JSON -Pjmh.fork=5 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/$name.json; echo "  $name exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
done; done
echo "REVERT-DONE $(date -u +%FT%TZ)"
