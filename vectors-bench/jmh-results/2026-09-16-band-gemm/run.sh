#!/usr/bin/env bash
# Band GEMM A/B runner. Usage (from the vectors repo root):
#   ./gradlew :vectors-bench:jmhJar
#   vectors-bench/jmh-results/2026-09-16-band-gemm/run.sh <label> [phases...]
# phases: bandwidth dequant ab-mt ab-st tile mapped crossover   (default: all but crossover)
# Env: THREADS (default: all logical CPUs), MAXBITS (default: unset = library default 256;
#      set 512 on AVX-512 hosts to test the wide species), EXTRA_JVM (extra JVM flags).
set -euo pipefail
LABEL=${1:?label, e.g. x86-avx512-epyc}; shift || true
PHASES=${*:-bandwidth dequant ab-mt ab-st tile mapped}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
JAR=$(ls "$ROOT"/vectors-bench/build/libs/vectors-bench-*-jmh.jar | head -1)
OUT="$ROOT/vectors-bench/jmh-results/2026-09-16-band-gemm/raw/$LABEL"
mkdir -p "$OUT"
JVM="--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC ${MAXBITS:+-Dvectors.maxBits=$MAXBITS} ${THREADS:+-Dvectors.gguf.threads=$THREADS} ${EXTRA_JVM:-}"

{
  echo "label=$LABEL date=$(date -u +%FT%TZ) commit=$(git -C "$ROOT" rev-parse --short HEAD)"
  uname -a
  (sysctl -n machdep.cpu.brand_string hw.ncpu hw.l1dcachesize hw.l2cachesize hw.memsize 2>/dev/null || lscpu) | tr '\n' ' '; echo
  (sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null || grep -m1 flags /proc/cpuinfo) | tr ' ' '\n' | grep -iE '^(avx|fma|asimd|sve)' | tr '\n' ' '; echo
  java -version 2>&1
  echo "jvm=$JVM"
} > "$OUT/host.txt"

run() { # name, jmh args...
  local name=$1; shift
  echo "== $name load-before: $(uptime)" | tee -a "$OUT/load.txt"
  java -jar "$JAR" "$@" -jvmArgs "$JVM" -rf json -rff "$OUT/$name.json" > "$OUT/$name.txt" 2>&1
  echo "== $name load-after:  $(uptime)" | tee -a "$OUT/load.txt"
}

for phase in $PHASES; do
  case $phase in
    bandwidth)
      run bandwidth-t1 MemoryBandwidthBenchmark -t 1 -f 1 -wi 2 -i 5 -w 2 -r 2
      run bandwidth-tall MemoryBandwidthBenchmark -t max -f 1 -wi 2 -i 5 -w 2 -r 2 ;;
    dequant)
      run dequant GgufBandDequantBenchmark -f 1 -wi 3 -i 5 -w 2 -r 2 ;;
    ab-mt)
      run ab-mt-existing GgufBandGemmAbBenchmark -p shape=1024x2048 -p batchSize=1,2,4,8,32 -f 1 -wi 3 -i 5 -w 2 -r 2
      run ab-mt-llm GgufBandGemmAbBenchmark -p shape=8192x2560,2560x8192 -p batchSize=1,512 -f 1 -wi 3 -i 5 -w 2 -r 2 ;;
    ab-st)
      JVM="$JVM -Dvectors.gguf.parallel=false"
      run ab-st-existing GgufBandGemmAbBenchmark -p shape=1024x2048 -p batchSize=1,8,32 -f 1 -wi 3 -i 5 -w 2 -r 2
      run ab-st-llm GgufBandGemmAbBenchmark -p shape=8192x2560,2560x8192 -p batchSize=1,512 -f 1 -wi 2 -i 3 -w 1 -r 1
      JVM="${JVM% -Dvectors.gguf.parallel=false}" ;;
    tile)
      for tile in 3x3 4x4; do
        JVM0=$JVM; JVM="$JVM -Dvectors.gguf.band.tile=$tile"
        run "tile-$tile" GgufBandGemmAbBenchmark -p kernel=BAND_F32 -p format=Q4_K,Q8_0 -p shape=8192x2560,2560x8192 -p batchSize=1,512 -f 1 -wi 3 -i 5 -w 2 -r 2
        JVM=$JVM0
      done ;;
    crossover)
      run crossover GgufBandGemmAbBenchmark -p shape=8192x2560,2560x8192 -p batchSize=1,2,4,8,16,32,64,128,512 -f 2 -wi 3 -i 5 -w 2 -r 2 ;;
    mapped)
      run ab-mt-mapped GgufBandGemmAbBenchmark -p storage=mapped -p format=Q4_K -p shape=8192x2560,2560x8192 -p batchSize=1,512 -f 1 -wi 3 -i 5 -w 2 -r 2 ;;
    *) echo "unknown phase $phase" >&2; exit 2 ;;
  esac
done
