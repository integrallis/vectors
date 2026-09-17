#!/usr/bin/env bash
# Pre-registration 4: layer-by-layer hidden-state probe of the integer and dispatch (band) arms
# against the float32 Transformers reference, on the three divergent base-arm prompts.
#
#   bash vectors-bench/jmh-results/2026-09-16-band-gemm/model-check/layer-probe/run-layer-probe.sh
#
# Expects, already on the host (from the model-check and reference-agreement runs):
#   $MC_WORK/models/models-bench/build/install/models-bench/lib   Models composite install
#   $MC_WORK/harness                                              compiled ModelCheck classes
#   $EVIDENCE/{window.json,cont-base-arm-integer.json,cont-base-arm-dispatch.json}
#   $PY (torch, transformers, gguf, numpy), $REF_DIR/reference_alora_case.py, $MODEL
# Writes everything under $OUT (default /opt/layerprobe/<UTC timestamp>/).
#
# Environment overrides: MC_WORK EVIDENCE PY REF_DIR MODEL OUT IDS EXTRA_JVM CONDITIONS
# Resume: re-run with the same OUT=... and finished Java arms / reference outputs are reused
# (set FORCE=1 to recompute them).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VECTORS_DIR="$(cd "$HERE/../../../../.." && pwd)"
MC_WORK="${MC_WORK:-/root/band-dispatch-model-check}"
LIB="${LIB:-$MC_WORK/models/models-bench/build/install/models-bench/lib}"
HARNESS="${HARNESS:-$MC_WORK/harness}"
EVIDENCE="${EVIDENCE:-$MC_WORK/evidence/20260917T022838Z}"
PY="${PY:-/opt/ref/venv/bin/python}"
REF_DIR="${REF_DIR:-/opt/ref}"
MODEL="${MODEL:-/opt/ref/granite-4.1-3b-Q4_K_M.gguf}"
GGUF_SHA=662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29
WINDOW_SHA=dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37
IDS="${IDS:-5737432bc3c5551400e51e9b,5705f09e75f01819005e77a4,57266193dd62a815002e832e}"
CONDITIONS="${CONDITIONS:-pipeline-cache fresh}"
EXTRA_JVM="${EXTRA_JVM:-}"
OUT="${OUT:-/opt/layerprobe/$(date -u +%Y%m%dT%H%M%SZ)}"
PROBE="$OUT/probe"
mkdir -p "$PROBE" "$OUT/classes"
if [ -n "${JAVA_HOME:-}" ]; then export PATH="$JAVA_HOME/bin:$PATH"; fi
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$OUT/run.log"; }
fail() { log "FATAL: $*"; exit 3; }

# --- Preconditions ------------------------------------------------------------------------------------
JAVA_VERSION_OUTPUT="$(java -version 2>&1)"
grep -q 'version "25' <<<"$JAVA_VERSION_OUTPUT" || fail "JDK 25 required on PATH"
for f in "$EVIDENCE/window.json" "$EVIDENCE/cont-base-arm-integer.json" "$EVIDENCE/cont-base-arm-dispatch.json" \
         "$REF_DIR/reference_alora_case.py" "$MODEL" "$HARNESS/ModelCheck.class" "$PY"; do
  [ -e "$f" ] || fail "missing $f"
done
[ "$(sha256sum "$EVIDENCE/window.json" | cut -d' ' -f1)" = "$WINDOW_SHA" ] || fail "window hash mismatch"
log "hashing model"
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$GGUF_SHA" ] || fail "model hash mismatch"
JAR_LISTING="$(unzip -l "$LIB"/vectors-core-*.jar)"
grep -q 'GgufBandGemm.class' <<<"$JAR_LISTING" || fail "vectors-core on the Models classpath has no GgufBandGemm"

{
  echo "vectorsCheckout=$(git -C "$VECTORS_DIR" rev-parse HEAD 2>/dev/null || echo unknown) (scripts only; kernels come from LIB)"
  echo "modelsInstall=$(git -C "$MC_WORK/models" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "vectorsCoreJar=$(sha256sum "$LIB"/vectors-core-*.jar)"
  echo "ids=$IDS conditions=$CONDITIONS extraJvm=$EXTRA_JVM evidence=$EVIDENCE model=$MODEL"
  uname -a; nproc; free -h 2>/dev/null || true; lscpu 2>/dev/null || true; uptime; echo "$JAVA_VERSION_OUTPUT"
  "$PY" -c 'import torch, transformers, numpy; print("torch", torch.__version__, "transformers", transformers.__version__, "numpy", numpy.__version__)'
} > "$OUT/host.txt" 2>&1
log "out $OUT"

# --- Compile ---------------------------------------------------------------------------------------------
javac --add-modules jdk.incubator.vector -nowarn -cp "$HARNESS:$LIB/*" -d "$OUT/classes" \
  "$HERE/LayerProbe.java" 2> "$OUT/javac.log" || { cat "$OUT/javac.log"; fail "javac failed"; }
cp "$HERE"/LayerProbe.java "$HERE"/reference_layers.py "$HERE"/compare_layers.py "$OUT/"

# --- Java arms: one fresh JVM per (condition, arm) -----------------------------------------------------------
JVM_BASE="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED $EXTRA_JVM"
for condition in $CONDITIONS; do
  extra_args=()
  if [ "$condition" = fresh ]; then extra_args=(--all-positions true); fi
  for arm in integer dispatch; do
    done_already=1
    for id in ${IDS//,/ }; do
      [ -s "$PROBE/$condition-$arm-$id.json" ] || done_already=0
    done
    if [ "$done_already" = 1 ] && [ "${FORCE:-0}" != 1 ]; then
      log "java condition=$condition arm=$arm: outputs present, reused"
      continue
    fi
    log "java condition=$condition arm=$arm load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || true)"
    # shellcheck disable=SC2086
    java $JVM_BASE -Dvectors.gguf.batchedMatmulKernel=$arm -Dvectors.gguf.batchedMatmulKernel.report=true \
      -cp "$OUT/classes:$HARNESS:$LIB/*" LayerProbe \
      --model "$MODEL" --window "$EVIDENCE/window.json" \
      --recorded "$EVIDENCE/cont-base-arm-integer.json" \
      --arm-continuations "$EVIDENCE/cont-base-arm-$arm.json" \
      --ids "$IDS" --condition "$condition" --out-dir "$PROBE" "${extra_args[@]}" \
      > "$OUT/java-$condition-$arm.log" 2>&1 || { tail -30 "$OUT/java-$condition-$arm.log"; fail "LayerProbe $condition $arm failed"; }
    grep -h "^LAYERPROBE" "$OUT/java-$condition-$arm.log" | tee -a "$OUT/run.log"
  done
done

# --- Reference ----------------------------------------------------------------------------------------------
probe_files=""
local_arms=""
for condition in $CONDITIONS; do
  probe_files="${probe_files:+$probe_files,}$condition-integer,$condition-dispatch"
  if [ "$condition" = fresh ]; then local_arms="integer,dispatch"; fi
done
ref_done=1
for id in ${IDS//,/ }; do [ -s "$PROBE/reference-$id.json" ] || ref_done=0; done
if [ "$ref_done" = 1 ] && [ "${FORCE:-0}" != 1 ]; then
  log "reference outputs present, reused"
else
log "reference (probe files $probe_files; local transfer: ${local_arms:-none})"
"$PY" "$HERE/reference_layers.py" --dir "$PROBE" --ids "$IDS" --gguf "$MODEL" \
  --reference-module-dir "$REF_DIR" --probe-files "$probe_files" --local-transfer "$local_arms" \
  > "$OUT/reference.log" 2>&1 || { tail -40 "$OUT/reference.log"; fail "reference_layers.py failed"; }
grep -h "^reference\|^WARNING" "$OUT/reference.log" | tee -a "$OUT/run.log"
fi

# --- Compare --------------------------------------------------------------------------------------------------
status=0
for condition in $CONDITIONS; do
  log "compare condition=$condition"
  set +e
  "$PY" "$HERE/compare_layers.py" --dir "$PROBE" --ids "$IDS" --condition "$condition" \
    --json "$OUT/compare-$condition.json" > "$OUT/summary-$condition.md" 2> "$OUT/compare-$condition.err"
  code=$?
  set -e
  cat "$OUT/summary-$condition.md" "$OUT/compare-$condition.err"
  if [ $code -ne 0 ]; then
    log "compare condition=$condition exited $code (misalignment)"
    status=$code
  fi
done
log "done: $OUT (exit $status)"
exit $status
