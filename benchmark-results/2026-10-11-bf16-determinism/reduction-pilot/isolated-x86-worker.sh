#!/bin/bash
set -euo pipefail
mkdir -p /work/out /opt/jdk
exec > >(tee -a /work/out/bootstrap.log /dev/console) 2>&1
remaining=$(( 1791770212 - $(date +%s) ))
if [ "$remaining" -le 0 ]; then shutdown -h now; exit 1; fi
nohup bash -c "sleep $remaining; shutdown -h now" >/work/out/deadline.log 2>&1 &
export RESULT_URI=s3://models-qual-077051030817/results/vectors-bf16-reduction-pilot-v1/x86/
finish() {
 rc=$?
 if [ "$rc" -ne 0 ]; then echo "FAILED $rc" > /work/out/STATUS; fi
 if command -v aws >/dev/null; then aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors || true; fi
 shutdown -h now
}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq unzip python3 ca-certificates
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -qo /tmp/awscli.zip -d /tmp
/tmp/aws/install
cd /work
aws s3 cp s3://models-qual-077051030817/payload/vectors-bf16-reduction-pilot-v1.tar.gz payload.tar.gz --only-show-errors
printf '%s  payload.tar.gz\n' 62c7510a83ca80b80ee9edd9af7bb6f40f68b4e601e98e74aacee98e778600b5 | sha256sum --check --status
tar xzf payload.tar.gz
sha256sum --check --status inputs.sha256
cp inputs.sha256 provenance.json out/
aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors
nohup bash -c 'while sleep 60; do aws s3 cp /work/out "$RESULT_URI" --recursive --only-show-errors; done' >/work/out/upload.log 2>&1 &
curl -fsSL 'https://api.adoptium.net/v3/assets/latest/25/hotspot?architecture=x64&image_type=jdk&os=linux&vendor=eclipse' -o out/jdk-release.json
python3 - <<'PYJDK'
import json,pathlib,urllib.request,hashlib
info=json.loads(pathlib.Path('out/jdk-release.json').read_text())[0]['binary']['package']
with urllib.request.urlopen(info['link']) as src,pathlib.Path('/tmp/jdk.tgz').open('wb') as dst:
 import shutil
 shutil.copyfileobj(src,dst)
assert hashlib.sha256(pathlib.Path('/tmp/jdk.tgz').read_bytes()).hexdigest()==info['checksum']
pathlib.Path('out/jdk.sha256').write_text(info['checksum']+'  /tmp/jdk.tgz\n')
PYJDK
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1
/opt/jdk/bin/java -version 2>out/java-version.txt
lscpu > out/lscpu.txt
uname -a > out/uname.txt
hostname > out/hostname.txt
ps aux > out/processes-before.txt
echo RUNNING_BF16_PAIRED > out/STATUS
python3 run-host.py
