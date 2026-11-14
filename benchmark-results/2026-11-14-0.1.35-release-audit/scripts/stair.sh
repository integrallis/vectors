#!/usr/bin/env bash
# Staircase bisect of AnnSearchBenchmark.searchHnsw across the first-parent commits between v0.1.34 and
# v0.1.35 that touch vectors-hnsw or vectors-db main sources. Each point is a detached worktree; round 1
# walks the chain oldest->newest, round 2 newest->oldest, so host drift cannot masquerade as a step.
set -uo pipefail
export JAVA_HOME=/usr/lib/jvm/temurin-25-jdk-amd64; export PATH="$JAVA_HOME/bin:$PATH"
OUT=$HOME/out/stair; mkdir -p $OUT $HOME/p
cd ~/vectors
lscpu > $HOME/out/cpu.txt; java -version > $HOME/out/java-version.txt 2>&1
POINTS=(v0.1.34 $(git log --first-parent --reverse --format=%h v0.1.34..v0.1.35 -- vectors-hnsw/src/main vectors-db/src/main) v0.1.35)
printf '%s\n' "${POINTS[@]}" > $OUT/points.txt
echo "=== ${#POINTS[@]} points: ${POINTS[*]}  $(date -u +%T)"
# build phase, 4 worktrees at a time
printf '%s\n' "${POINTS[@]}" | xargs -P 4 -I{} bash -c 'cd ~/vectors && git worktree add -q --detach ~/p/{} {} && cd ~/p/{} && ./gradlew :vectors-bench:compileJmhJava --no-daemon -q > ~/p/{}.build.log 2>&1; echo "built {} exit=$?"'
echo "=== builds done $(date -u +%T)"
run_point() { # $1=index $2=point $3=round
  local idx=$1 p=$2 r=$3 name
  name=$(printf '%02d' $idx)-$p-r$r
  (cd ~/p/$p && rm -rf vectors-bench/build/results/jmh && ./gradlew :vectors-bench:jmh -Pjmh.includes='AnnSearchBenchmark\.searchHnsw$' -Pjmh.resultFormat=JSON -Pjmh.fork=2 -Pjmh.warmup=3 -Pjmh.iterations=5 --no-daemon > $OUT/$name.log 2>&1; rc=$?; f=vectors-bench/build/results/jmh/results.json; [ -s $f ] && cp $f $OUT/$name.json; echo "  $name exit=$rc json=$([ -s $f ] && echo yes || echo NO) $(date -u +%T)")
}
n=${#POINTS[@]}
for i in $(seq 0 $((n-1))); do run_point $i ${POINTS[$i]} 1; done
for i in $(seq $((n-1)) -1 0); do run_point $i ${POINTS[$i]} 2; done
echo "STAIR-DONE $(date -u +%FT%TZ)"
