#!/usr/bin/env bash
# SIMD K-quant dequant A/B runner. Usage (from the vectors repo root):
#   ./gradlew :vectors-bench:jmhJar
#   vectors-bench/jmh-results/2026-09-17-kquant-dequant/run.sh <label> [phases...]
# phases: dequant ab ab-st   (default: dequant ab)
#   dequant  GgufBandDequantBenchmark, single thread, Q4_K/Q6_K x Granite FFN shapes x every arm
#            (arms selected explicitly inside one JVM per fork).
#   ab       GgufBandGemmAbBenchmark, executor default (multi-threaded): INTEGER once as reference,
#            then BAND_F32 once per dequant arm (-Dvectors.gguf.band.dequant, one JMH run per arm),
#            Q4_K/Q6_K x 8192x2560,2560x8192 x batch 1,2,4,8,512. Each fork asserts it runs the
#            labelled arm (param dequant) and prints bandDequant=... in the .txt output.
#   ab-st    as ab with -Dvectors.gguf.parallel=false (not part of the decision rule).
# Env: MAXBITS (unset = library default 256; 512 on AVX-512 hosts), THREADS (GGUF executor threads),
#      ARMS (default "scalar simd-byte simd-int simd-split"), ROUNDS (default 1; arms are rotated
#      each round and files get a -r<N> suffix), BATCHES (default 1,2,4,8,512),
#      FORKS WI I W R (JMH forks, warmup/measurement iterations and seconds; default 2 3 5 2 2),
#      EXTRA_JVM (extra JVM flags).
set -euo pipefail
LABEL=${1:?label, e.g. genoa-avx512-512bit}; shift || true
PHASES=${*:-dequant ab}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
JAR=$(ls "$ROOT"/vectors-bench/build/libs/vectors-bench-*-jmh.jar | head -1)
OUT="$ROOT/vectors-bench/jmh-results/2026-09-17-kquant-dequant/raw/$LABEL"
mkdir -p "$OUT"
ARMS=${ARMS:-scalar simd-byte simd-int simd-split}
ROUNDS=${ROUNDS:-1}
BATCHES=${BATCHES:-1,2,4,8,512}
FORKS=${FORKS:-2}; WI=${WI:-3}; I=${I:-5}; W=${W:-2}; R=${R:-2}
JMH_ITER=(-f "$FORKS" -wi "$WI" -i "$I" -w "$W" -r "$R")
JVM="--add-modules jdk.incubator.vector -Xmx8g -Xms8g -XX:+AlwaysPreTouch -XX:+UseG1GC ${MAXBITS:+-Dvectors.maxBits=$MAXBITS} ${THREADS:+-Dvectors.gguf.threads=$THREADS} ${EXTRA_JVM:-}"
SHAPES=8192x2560,2560x8192

{
  echo "label=$LABEL date=$(date -u +%FT%TZ) commit=$(git -C "$ROOT" rev-parse --short HEAD) dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  echo "phases=$PHASES arms=$ARMS rounds=$ROUNDS batches=$BATCHES jmh=${JMH_ITER[*]}"
  uname -a
  (sysctl -n machdep.cpu.brand_string hw.ncpu hw.l1dcachesize hw.l2cachesize hw.memsize 2>/dev/null || lscpu) | tr '\n' ' '; echo
  (sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null || grep -m1 flags /proc/cpuinfo) | tr ' ' '\n' | grep -iE '^(avx|fma|asimd|sve)' | tr '\n' ' '; echo
  java -version 2>&1
  echo "jvm=$JVM"
} > "$OUT/host.txt"

run() { # name, jmh args... (uses $JVM)
  local name=$1; shift
  echo "== $name load-before: $(uptime)" | tee -a "$OUT/load.txt"
  java -jar "$JAR" "$@" -jvmArgs "$JVM" -rf json -rff "$OUT/$name.json" > "$OUT/$name.txt" 2>&1
  echo "== $name load-after:  $(uptime)" | tee -a "$OUT/load.txt"
  grep -m1 -E "bandDequant=|band-dequant " "$OUT/$name.txt" | sed 's/^/   /' || true
}

rotate() { # round index, words... -> words rotated left by round
  local n=$1; shift; local a=("$@"); local k=$(( n % ${#a[@]} ))
  echo "${a[@]:k}" "${a[@]:0:k}"
}

ab_phase() { # suffix, extra jvm flags
  local suffix=$1 extra=$2 round arm base_jvm=$JVM
  for ((round = 1; round <= ROUNDS; round++)); do
    local tag=""; [ "$ROUNDS" -gt 1 ] && tag="-r$round"
    JVM="$base_jvm $extra"
    run "ab$suffix-integer$tag" GgufBandGemmAbBenchmark -p kernel=INTEGER -p format=Q4_K,Q6_K \
      -p shape=$SHAPES -p batchSize=$BATCHES -p dequant=any "${JMH_ITER[@]}"
    # shellcheck disable=SC2046
    for arm in $(rotate $((round - 1)) $ARMS); do
      JVM="$base_jvm $extra -Dvectors.gguf.band.dequant=$arm"
      run "ab$suffix-band-$arm$tag" GgufBandGemmAbBenchmark -p kernel=BAND_F32 -p format=Q4_K,Q6_K \
        -p shape=$SHAPES -p batchSize=$BATCHES -p dequant=$arm "${JMH_ITER[@]}"
    done
    JVM=$base_jvm
  done
}

for phase in $PHASES; do
  case $phase in
    dequant)
      run dequant GgufBandDequantBenchmark -p format=Q4_K,Q6_K -p shape=$SHAPES \
        -p dequant=$(echo $ARMS | tr ' ' ',') "${JMH_ITER[@]}" ;;
    ab) ab_phase "" "" ;;
    ab-st) ab_phase "-st" "-Dvectors.gguf.parallel=false" ;;
    *) echo "unknown phase $phase" >&2; exit 2 ;;
  esac
done
