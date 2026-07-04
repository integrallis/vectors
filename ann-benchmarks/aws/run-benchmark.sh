#!/usr/bin/env bash
#
# Copyright 2025-2026 Integrallis Software, LLC. Apache-2.0.
#
# Provision-free ANN-Benchmarks runner for `vectors`. Builds vectors-server,
# stages the adapter into an ann-benchmarks checkout, runs the harness for the
# requested datasets/algorithms, and optionally uploads results to S3.
#
# The ANN-Benchmarks suite is single-node by design (that is what makes the
# numbers comparable), so this is meant to run on ONE host — e.g. a single EC2
# instance provisioned by the Terraform in this directory, or any Ubuntu 22.04
# box you already have. No Kubernetes.
#
# Config is via environment variables (all optional):
#   VECTORS_REPO   git URL of the vectors repo (default: integrallis/vectors)
#   VECTORS_REF    branch/tag/sha to build               (default: main)
#   ANN_REPO       ann-benchmarks git URL                (default: erikbern/ann-benchmarks)
#   ANN_REF        ann-benchmarks ref                    (default: main)
#   DATASETS       space-separated ann-benchmarks datasets (default: sift-128-euclidean)
#                  e.g. "sift-128-euclidean glove-100-angular" for a multi-dataset sweep
#                  (DATASET, singular, is still accepted for backward compatibility)
#   ALGORITHMS     space-separated algorithms to run     (default: "vectors")
#                  e.g. "vectors hnswlib faiss qdrant" for a head-to-head
#   RUN_COUNT      runs per config (best is kept)        (default: 3)
#   WORKDIR        scratch dir                           (default: $HOME/vectors-bench)
#   S3_RESULTS     optional s3://bucket/prefix to upload results/ to
set -euo pipefail

VECTORS_REPO="${VECTORS_REPO:-https://github.com/integrallis/vectors.git}"
VECTORS_REF="${VECTORS_REF:-main}"
ANN_REPO="${ANN_REPO:-https://github.com/erikbern/ann-benchmarks.git}"
ANN_REF="${ANN_REF:-main}"
DATASETS="${DATASETS:-${DATASET:-sift-128-euclidean}}"
ALGORITHMS="${ALGORITHMS:-vectors}"
RUN_COUNT="${RUN_COUNT:-3}"
WORKDIR="${WORKDIR:-$HOME/vectors-bench}"
S3_RESULTS="${S3_RESULTS:-}"

log() { echo "[run-benchmark] $*"; }

install_deps() {
  if command -v docker >/dev/null 2>&1 && command -v java >/dev/null 2>&1; then
    return
  fi
  log "installing docker, git, python3, Temurin JDK 25"
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -y
  sudo apt-get install -y docker.io git python3 python3-pip python3-venv wget ca-certificates gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public |
    sudo gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg
  . /etc/os-release
  echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb ${VERSION_CODENAME} main" |
    sudo tee /etc/apt/sources.list.d/adoptium.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y temurin-25-jdk
  sudo systemctl enable --now docker || true
  sudo usermod -aG docker "$USER" || true
}

# Fail fast if the harness won't be able to reach the Docker daemon. The usermod above does not
# take effect in the current shell, so a standalone (non-root) run needs the invoking user to
# already be in the docker group (re-login) or to run under sudo. Terraform user_data runs as root,
# where this is a no-op.
require_docker_ready() {
  if ! docker ps >/dev/null 2>&1; then
    log "ERROR: cannot talk to the Docker daemon as $(id -un)."
    log "  ann-benchmarks shells out to 'docker' directly, so this must work without sudo."
    log "  Fix: log out/in after the docker-group change, or run this script as root."
    exit 1
  fi
}

# Reproducibility manifest — the audit requires committed hardware/JDK/dataset/params/SHA metadata
# for any published benchmark. Written into results/ so it travels with the raw output to S3.
capture_manifest() {
  local out="$WORKDIR/ann-benchmarks/results/vectors-bench-manifest.txt"
  mkdir -p "$(dirname "$out")"
  local vectors_sha ann_sha
  vectors_sha="$(git -C "$WORKDIR/vectors" rev-parse HEAD 2>/dev/null || echo unknown)"
  ann_sha="$(git -C "$WORKDIR/ann-benchmarks" rev-parse HEAD 2>/dev/null || echo unknown)"
  {
    echo "# vectors ANN-Benchmarks run manifest"
    echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "datasets=$DATASETS"
    echo "algorithms=$ALGORITHMS"
    echo "run_count=$RUN_COUNT"
    echo "vectors_repo=$VECTORS_REPO"
    echo "vectors_ref=$VECTORS_REF"
    echo "vectors_sha=$vectors_sha"
    echo "ann_benchmarks_repo=$ANN_REPO"
    echo "ann_benchmarks_ref=$ANN_REF"
    echo "ann_benchmarks_sha=$ann_sha"
    echo "jvm_flags=--add-modules jdk.incubator.vector"
    echo "--- java ---"; java -version 2>&1 || echo "java unavailable"
    echo "--- os ---"; uname -a; (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || true
    echo "--- cpu ---"; lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    echo "--- memory ---"; free -h 2>/dev/null || true
    echo "--- ec2 instance-type (if on EC2) ---"
    curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "n/a"
  } > "$out"
  log "wrote reproducibility manifest -> $out"
}

build_vectors() {
  rm -rf "$WORKDIR/vectors"
  # VECTORS_LOCAL_DIR: build from a pre-staged checkout instead of cloning — for a private repo
  # (rsync the source over rather than putting a token on the box) or local dev. Java is
  # cross-platform, so the dist is (re)built on this host regardless.
  if [ -n "${VECTORS_LOCAL_DIR:-}" ]; then
    log "using pre-staged vectors checkout at $VECTORS_LOCAL_DIR"
    cp -r "$VECTORS_LOCAL_DIR" "$WORKDIR/vectors"
  else
    log "cloning vectors @ ${VECTORS_REF}"
    git clone --depth 1 --branch "$VECTORS_REF" "$VECTORS_REPO" "$WORKDIR/vectors" ||
      git clone "$VECTORS_REPO" "$WORKDIR/vectors"
    (cd "$WORKDIR/vectors" && git checkout "$VECTORS_REF")
  fi
  log "building vectors-server dist"
  (cd "$WORKDIR/vectors" && ./gradlew :vectors-server:installDist)
}

stage_adapter() {
  log "staging ann-benchmarks @ ${ANN_REF} with the vectors adapter"
  rm -rf "$WORKDIR/ann-benchmarks"
  git clone --depth 1 --branch "$ANN_REF" "$ANN_REPO" "$WORKDIR/ann-benchmarks" ||
    git clone "$ANN_REPO" "$WORKDIR/ann-benchmarks"
  local src="$WORKDIR/vectors/ann-benchmarks"
  local dst="$WORKDIR/ann-benchmarks/ann_benchmarks/algorithms/vectors"
  mkdir -p "$dst"
  cp "$src/module.py" "$src/config.yml" "$src/Dockerfile" "$dst/"
  rm -rf "$dst/vectors-server-dist"
  cp -r "$WORKDIR/vectors/vectors-server/build/install/vectors-server" "$dst/vectors-server-dist"
  # in-process jpype variant as a sibling algorithm
  local dstj="$WORKDIR/ann-benchmarks/ann_benchmarks/algorithms/vectors_jpype"
  mkdir -p "$dstj"
  # The harness imports "<module>.module", so the file must be module.py (not module_jpype.py).
  cp "$src/module_jpype.py" "$dstj/module.py"
  cp "$src/config.jpype.yml" "$dstj/config.yml"
  cp "$src/Dockerfile.jpype" "$dstj/Dockerfile"
  cp -r "$dst/vectors-server-dist" "$dstj/vectors-server-dist"
}

run_harness() {
  cd "$WORKDIR/ann-benchmarks"
  # Ubuntu 24.04 (noble) ships a PEP-668 externally-managed Python, so a system-wide
  # `pip install` is refused. Use a dedicated venv for the host-side harness (the algorithms
  # themselves still run in their own Docker containers).
  local venv="$WORKDIR/annb-venv"
  [ -d "$venv" ] || python3 -m venv "$venv"
  local py="$venv/bin/python"
  "$py" -m pip install --upgrade pip >/dev/null
  "$py" -m pip install -r requirements.txt
  for algo in $ALGORITHMS; do
    log "install.py --algorithm $algo"
    "$py" install.py --algorithm "$algo"
  done
  local algo_args=()
  for algo in $ALGORITHMS; do algo_args+=(--algorithm "$algo"); done
  for dataset in $DATASETS; do
    log "run.py --dataset $dataset ${algo_args[*]} --runs $RUN_COUNT"
    "$py" run.py --dataset "$dataset" "${algo_args[@]}" --runs "$RUN_COUNT"
    log "plot.py --dataset $dataset"
    "$py" plot.py --dataset "$dataset" -o "results-${dataset}.png" || true
  done
}

upload_results() {
  [ -n "$S3_RESULTS" ] || return 0
  # S3_ENDPOINT lets results go to any S3-compatible store — e.g. Cloudflare R2
  # (S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com) with AWS_ACCESS_KEY_ID /
  # AWS_SECRET_ACCESS_KEY set to the R2 token. Empty = real AWS S3.
  local ep=()
  [ -n "${S3_ENDPOINT:-}" ] && ep=(--endpoint-url "$S3_ENDPOINT")
  log "uploading results to $S3_RESULTS${S3_ENDPOINT:+ (endpoint $S3_ENDPOINT)}"
  aws "${ep[@]}" s3 cp --recursive "$WORKDIR/ann-benchmarks/results" "$S3_RESULTS/results"
  for dataset in $DATASETS; do
    aws "${ep[@]}" s3 cp "$WORKDIR/ann-benchmarks/results-${dataset}.png" "$S3_RESULTS/" 2>/dev/null || true
  done
}

mkdir -p "$WORKDIR"
install_deps
require_docker_ready
build_vectors
stage_adapter
run_harness
capture_manifest
upload_results
log "done. results in $WORKDIR/ann-benchmarks/results${S3_RESULTS:+ and $S3_RESULTS}"
