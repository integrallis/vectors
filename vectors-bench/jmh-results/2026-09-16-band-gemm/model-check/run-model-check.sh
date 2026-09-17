#!/usr/bin/env bash
# Model-level check for the band/integer batch dispatch (pre-registration 2), one command:
#
#   bash vectors-bench/jmh-results/2026-09-16-band-gemm/model-check/run-model-check.sh
#
# Runs on an idle x86-64 Linux host with JDK 25 on PATH (or JAVA_HOME), git, curl, python3, unzip,
# sha256sum. Builds Models at a pinned commit against THIS Vectors checkout with --include-build,
# then measures, dispatch off (integer) vs on (dispatch), fresh JVM per measurement:
#   1. pure-Java Granite 4.1 3B Q4_K_M prefill tok/s, prompt ~2048 tokens, context 2048
#      (Models `models-bench profile-prefill`), PREFILL_REPS interleaved runs per arm, median;
#      gate: dispatch median >= 1.10 x integer median.
#   2. greedy 64-token continuations of the first 20 squad-v2-dev cases of the frozen window v2,
#      base arm (the activated-answerability runner's base-arm prompt; no adapter loaded);
#      gate: identical token-fragment sequences in >= 19 of 20.
#      Supplementary, not gating: the same cases without the one-word instruction ("open").
# Prints both results and writes everything under $WORK/evidence/<timestamp>/.
#
# Environment overrides:
#   WORK            work root (default: $HOME/band-dispatch-model-check)
#   MODEL           existing granite-4.1-3b-Q4_K_M.gguf (default: downloaded + hash-checked)
#   MODELS_COMMIT   Models revision (default: origin/main 0.3.41, 167a8abd)
#   PREFILL_REPS    runs per arm (default 5)
#   TARGET_TOKENS   prefill prompt tokens (default 2040, context 2048)
#   EXTRA_JVM       extra JVM flags for BOTH arms, e.g. "-Dvectors.maxBits=512"
#   ADAPTER_DIR     optional answerability aLoRA directory: if set, the Models runner dumps its own
#                   base-arm prompts and the script checks byte parity with this harness's prompts
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VECTORS_DIR="${VECTORS_DIR:-$(cd "$HERE/../../../.." && pwd)}"
WORK="${WORK:-$HOME/band-dispatch-model-check}"
MODELS_COMMIT="${MODELS_COMMIT:-167a8abdd662af8d89a821a1bf2d23980e1092c5}"
WINDOW_COMMIT=cb2f42624d6cb75e8caa9fbe1715233ea7148eb2
WINDOW_PATH=benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/qualification-window-v2.json
WINDOW_SHA=dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37
GGUF_URL='https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/ab4701481089b58a082ef63cc1cee738887293ff/granite-4.1-3b-Q4_K_M.gguf'
GGUF_SHA=662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29
MODEL="${MODEL:-$WORK/store/granite-4.1-3b-Q4_K_M.gguf}"
PREFILL_REPS="${PREFILL_REPS:-5}"
TARGET_TOKENS="${TARGET_TOKENS:-2040}"
CONTEXT=2048
EXTRA_JVM="${EXTRA_JVM:-}"
EV="$WORK/evidence/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EV" "$WORK/store"
[ -n "${JAVA_HOME:-}" ] && export PATH="$JAVA_HOME/bin:$PATH"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$EV/run.log"; }

java -version 2>&1 | grep -q 'version "25' || { echo "JDK 25 required on PATH" >&2; exit 2; }
for tool in git curl python3 unzip sha256sum javac; do command -v $tool >/dev/null || { echo "missing $tool" >&2; exit 2; }; done
test -z "$(git -C "$VECTORS_DIR" status --porcelain)" || { echo "vectors checkout is dirty: $VECTORS_DIR" >&2; exit 2; }

{
  echo "vectors=$(git -C "$VECTORS_DIR" rev-parse HEAD) models=$MODELS_COMMIT window=$WINDOW_COMMIT"
  echo "reps=$PREFILL_REPS targetTokens=$TARGET_TOKENS context=$CONTEXT extraJvm=$EXTRA_JVM"
  uname -a; nproc; free -h 2>/dev/null || true; lscpu 2>/dev/null || true; uptime; java -version 2>&1
} > "$EV/host.txt"
log "evidence $EV"

# --- Models at the pinned commit, built against this Vectors checkout -----------------------------
MODELS="$WORK/models"
[ -d "$MODELS/.git" ] || git clone --quiet https://github.com/integrallis/models.git "$MODELS"
git -C "$MODELS" fetch --quiet origin "$MODELS_COMMIT" "$WINDOW_COMMIT"
git -C "$MODELS" switch --detach --quiet "$MODELS_COMMIT"
test -z "$(git -C "$MODELS" status --porcelain)"
log "build models $MODELS_COMMIT --include-build $VECTORS_DIR"
( cd "$MODELS" && ./gradlew --include-build "$VECTORS_DIR" :models-bench:installDist -x test --console=plain ) \
  > "$EV/build.log" 2>&1
INSTALL="$MODELS/models-bench/build/install/models-bench"
LIB="$INSTALL/lib"
unzip -l "$LIB"/vectors-core-*.jar | grep -q 'GgufBandGemm.class' \
  || { log "FATAL: vectors-core on the Models classpath is not this branch (no GgufBandGemm)"; exit 3; }
log "composite ok: $(ls "$LIB" | grep vectors-core)"

# --- Pinned inputs -----------------------------------------------------------------------------------
if [ ! -f "$MODEL" ]; then
  log "download model"
  curl -fsSL --retry 5 -o "$MODEL.part" "$GGUF_URL" && mv "$MODEL.part" "$MODEL"
fi
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$GGUF_SHA" ] || { log "FATAL: model hash mismatch"; exit 3; }
git -C "$MODELS" show "$WINDOW_COMMIT:$WINDOW_PATH" > "$EV/window.json"
[ "$(sha256sum "$EV/window.json" | cut -d' ' -f1)" = "$WINDOW_SHA" ] || { log "FATAL: window hash mismatch"; exit 3; }
log "inputs verified model=$GGUF_SHA window=$WINDOW_SHA"

mkdir -p "$WORK/harness"
javac --add-modules jdk.incubator.vector -nowarn -cp "$LIB/*" -d "$WORK/harness" "$HERE/ModelCheck.java" 2> "$EV/javac.log"
JVM_BASE="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED $EXTRA_JVM"
arm_flags() { echo "-Dvectors.gguf.batchedMatmulKernel=$1 -Dvectors.gguf.batchedMatmulKernel.report=true"; }
harness() { # arm, args...
  local arm=$1; shift
  # shellcheck disable=SC2046
  java $JVM_BASE $(arm_flags "$arm") -cp "$WORK/harness:$LIB/*" ModelCheck "$@"
}

# --- 1. Prefill tok/s ----------------------------------------------------------------------------------
python3 - "$EV/window.json" "$EV/prefill-source.txt" <<'PY'
import json, sys
window = json.load(open(sys.argv[1]))
suite = next(s for s in window["suites"] if s["name"] == "squad-v2-dev")
seen, texts = set(), []
for case in suite["cases"]:
    for doc in case["documents"]:
        if doc["text"] not in seen:
            seen.add(doc["text"]); texts.append(doc["text"])
open(sys.argv[2], "w").write("\n\n".join(texts))
PY
harness integer calibrate --model "$MODEL" --text "$EV/prefill-source.txt" --target-tokens "$TARGET_TOKENS" \
  --out "$EV/prefill-prompt.txt" 2>&1 | tee "$EV/calibrate.log" | grep calibrated | tee -a "$EV/run.log"

for rep in $(seq 1 "$PREFILL_REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then order="integer dispatch"; else order="dispatch integer"; fi
  for arm in $order; do
    log "prefill rep=$rep arm=$arm load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"
    JAVA_OPTS="$EXTRA_JVM $(arm_flags "$arm")" "$INSTALL/bin/models-bench" profile-prefill \
      --model "$MODEL" --prompt-file "$EV/prefill-prompt.txt" --context "$CONTEXT" --warmups 2 \
      --output "$EV/prefill-$arm-$rep.jfr" > "$EV/prefill-$arm-$rep.log" 2>&1
    grep -h "prefill profile" "$EV/prefill-$arm-$rep.log" | tee -a "$EV/run.log"
  done
done

# --- 2. Greedy continuations ----------------------------------------------------------------------------
for variant in base-arm open; do
  for arm in integer dispatch; do
    log "continuations variant=$variant arm=$arm"
    harness "$arm" continuations --model "$MODEL" --window "$EV/window.json" --suite squad-v2-dev \
      --limit 20 --max-tokens 64 --variant "$variant" --out "$EV/cont-$variant-$arm.json" \
      > "$EV/cont-$variant-$arm.log" 2>&1
  done
done

if [ -n "${ADAPTER_DIR:-}" ]; then
  log "runner prompt dump for parity (adapter $ADAPTER_DIR)"
  "$INSTALL/bin/models-bench" activated-answerability --model "$MODEL" --adapter "$ADAPTER_DIR" \
    --models-revision "$MODELS_COMMIT" --window "$EV/window.json" --suite squad-v2-dev --arm base \
    --limit 20 --dump-prompts "$EV/runner-prompts" --report "$EV/runner-report.json" \
    > "$EV/runner-dump.log" 2>&1 || log "runner prompt dump failed; see runner-dump.log"
fi

python3 "$HERE/compare.py" "$EV" | tee "$EV/summary.md"
log "done: $EV"
