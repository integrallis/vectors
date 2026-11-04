#!/usr/bin/env bash
# Measures GGUF row-executor dispatch cost: Phaser barrier vs generation/countdown publication.
# Self-contained: one JMH jar from S3, no repository on the box.
set -uo pipefail
export HOME=/root
BUCKET=models-qual-077051030817
RUN=2026-11-03-gguf-stage
PAYLOAD=gguf-barrier-ab-0002.jar
OUT=/work/out
LOG=/work/worker.log
STATUS=INCOMPLETE

mkdir -p /work "$OUT" /opt/jdk 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

finish() {
  echo "$STATUS" > "$OUT/STATUS"
  aws s3 cp "$OUT/STATUS" "s3://$BUCKET/results/$RUN/STATUS" --only-show-errors 2>/dev/null
  aws s3 cp "$LOG" "s3://$BUCKET/results/$RUN/worker.log" --only-show-errors 2>/dev/null
  say "final status $STATUS"
  shutdown -h now
}
trap finish EXIT

# A response ARRIVING is the test: s3 root answers an unauthenticated HEAD with 307, so curl -f
# would call a healthy box dead. 000 means nothing answered.
EGRESS_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 https://s3.amazonaws.com 2>/dev/null)
if [ "$EGRESS_CODE" = "000" ] || [ -z "$EGRESS_CODE" ]; then
  STATUS=NO_EGRESS; say "no outbound HTTPS (curl code=$EGRESS_CODE)"; exit 1
fi
say "egress ok (s3 root http=$EGRESS_CODE)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || say "apt-get update failed (continuing)"
apt-get install -y -qq unzip python3 || say "apt-get install failed (continuing)"
# The bundled installer is the primary path: this AMI has no awscli candidate in apt.
if ! command -v aws >/dev/null 2>&1; then
  say "installing aws cli from the bundled installer"
  if curl -sfL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip; then
    (cd /tmp && unzip -qo awscli.zip && ./aws/install --update >/dev/null 2>&1); hash -r
  fi
fi
command -v aws >/dev/null 2>&1 || { STATUS=NO_AWS_CLI; exit 1; }

# Machine specs belong beside the result, uploaded now rather than at the end: a box that dies
# mid-run still has to say what it was.
{ echo "== uname =="; uname -srm
  echo "== cpu =="; lscpu
  echo "== meminfo =="; head -3 /proc/meminfo
  echo "== instance =="
  TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null)
  for f in instance-type instance-id placement/availability-zone; do
    printf '%s=%s\n' "$f" "$(curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$f" 2>/dev/null)"
  done
} > "$OUT/machine.txt" 2>&1
aws s3 cp "$OUT/machine.txt" "s3://$BUCKET/results/$RUN/machine.txt" --only-show-errors 2>/dev/null
say "specs uploaded: $(grep -m1 'Model name' "$OUT/machine.txt" || true)"

curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz \
  || { STATUS=NO_JDK_DOWNLOAD; exit 1; }
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 || { STATUS=NO_JDK_UNPACK; exit 1; }
export JAVA_HOME=/opt/jdk PATH=/opt/jdk/bin:$PATH
[ -x /opt/jdk/bin/java ] || { STATUS=NO_JDK; exit 1; }
say "jdk $(java -version 2>&1 | head -1)"
java -version > "$OUT/java-version.txt" 2>&1

cd /work
aws s3 cp "s3://$BUCKET/payload/$PAYLOAD" ./bench.jar --only-show-errors || { STATUS=NO_PAYLOAD; exit 1; }
[ -s ./bench.jar ] || { STATUS=NO_PAYLOAD; exit 1; }
say "payload $(stat -c %s ./bench.jar) bytes"

# The whole sweep in one JVM invocation per fork; JMH owns forking so each parameter
# combination gets fresh JVMs rather than inheriting a warmed pool.
say "running sweep"
java -jar ./bench.jar GgufStagePlanBenchmark \
  -rf json -rff "$OUT/dispatch.json" -foe false 2>&1 | tail -400
RC=${PIPESTATUS[0]}
say "jmh exit $RC"

aws s3 cp "$OUT/dispatch.json" "s3://$BUCKET/results/$RUN/dispatch.json" --only-show-errors 2>/dev/null
if [ -s "$OUT/dispatch.json" ]; then
  STATUS=OK
  say "results uploaded"
else
  STATUS=NO_RESULTS
fi
exit 0
