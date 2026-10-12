#!/bin/bash
set -euo pipefail
mkdir -p /work/out /opt/jdk
exec > >(tee -a /work/out/bootstrap.log /dev/console) 2>&1
remaining=$(( 1791826384 - $(date +%s) ))
if [ "$remaining" -le 0 ]; then shutdown -h now; exit 1; fi
nohup bash -c "sleep $remaining; shutdown -h now" >/work/out/deadline.log 2>&1 &
export RESULT_URI=s3://models-qual-077051030817/results/vectors-bf16-masked-pair-v1/arm/
finish() {
 rc=$?
 if [ "$rc" -ne 0 ]; then echo "FAILED $rc" > /work/out/STATUS; fi
 if command -v aws >/dev/null; then aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors || true; fi
 shutdown -h now
}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl unzip python3 ca-certificates build-essential cmake time
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip -o /tmp/awscli.zip
unzip -qo /tmp/awscli.zip -d /tmp
/tmp/aws/install
cd /work
aws s3 cp s3://models-qual-077051030817/payload/vectors-bf16-masked-pair-v1.tar.gz payload.tar.gz --only-show-errors
printf '%s  payload.tar.gz
' 835b9887074cf45fd1bb00f843dbaec4fce6f15e3d8deb1dd08ebc630f08111c | sha256sum --check --status
tar xzf payload.tar.gz
sha256sum --check --status inputs.sha256
cp inputs.sha256 provenance.json out/
aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors
nohup bash -c 'while sleep 60; do aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors; done' >out/upload.log 2>&1 &
curl -fL --retry 3 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_aarch64_linux_hotspot_25.0.3_9.tar.gz' -o /tmp/jdk.tgz
printf '%s  /tmp/jdk.tgz
' 3e4287cb98870ba824ed698854bdc27cff984254caf66dd12cc291e7bfdde26b | sha256sum --check --status
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1
curl -fsSL https://sh.rustup.rs -o /tmp/rustup.sh
sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.96.0
/opt/jdk/bin/java -version 2>out/java-version.txt
/root/.cargo/bin/rustc --version >out/rust-version.txt
lscpu >out/lscpu.txt
uname -a >out/uname.txt
ps aux >out/processes-before.txt
python3 run-host.py
