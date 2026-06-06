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
#   DATASET        ann-benchmarks dataset                (default: sift-128-euclidean)
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
DATASET="${DATASET:-sift-128-euclidean}"
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

build_vectors() {
  log "building vectors-server @ ${VECTORS_REF}"
  rm -rf "$WORKDIR/vectors"
  git clone --depth 1 --branch "$VECTORS_REF" "$VECTORS_REPO" "$WORKDIR/vectors" ||
    git clone "$VECTORS_REPO" "$WORKDIR/vectors"
  (cd "$WORKDIR/vectors" && git checkout "$VECTORS_REF" && ./gradlew :vectors-server:installDist)
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
  cp "$src/module_jpype.py" "$dstj/"
  cp "$src/config.jpype.yml" "$dstj/config.yml"
  cp "$src/Dockerfile.jpype" "$dstj/Dockerfile"
  cp -r "$dst/vectors-server-dist" "$dstj/vectors-server-dist"
}

run_harness() {
  cd "$WORKDIR/ann-benchmarks"
  python3 -m pip install --user -r requirements.txt
  for algo in $ALGORITHMS; do
    log "install.py --algorithm $algo"
    python3 install.py --algorithm "$algo"
  done
  local algo_args=()
  for algo in $ALGORITHMS; do algo_args+=(--algorithm "$algo"); done
  log "run.py --dataset $DATASET ${algo_args[*]} --runs $RUN_COUNT"
  python3 run.py --dataset "$DATASET" "${algo_args[@]}" --runs "$RUN_COUNT"
  log "plot.py --dataset $DATASET"
  python3 plot.py --dataset "$DATASET" -o "results-${DATASET}.png" || true
}

upload_results() {
  [ -n "$S3_RESULTS" ] || return 0
  log "uploading results to $S3_RESULTS"
  aws s3 cp --recursive "$WORKDIR/ann-benchmarks/results" "$S3_RESULTS/results"
  aws s3 cp "$WORKDIR/ann-benchmarks/results-${DATASET}.png" "$S3_RESULTS/" 2>/dev/null || true
}

mkdir -p "$WORKDIR"
install_deps
build_vectors
stage_adapter
run_harness
upload_results
log "done. results in $WORKDIR/ann-benchmarks/results${S3_RESULTS:+ and $S3_RESULTS}"
