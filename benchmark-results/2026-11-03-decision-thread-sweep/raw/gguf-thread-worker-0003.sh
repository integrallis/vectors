#!/usr/bin/env bash
# Why is one typed decision 2.5 s for a 1.5B on 32 threads? Sweep both pools' thread counts.
set -uo pipefail
export HOME=/root
BUCKET=models-qual-077051030817
RUN=2026-11-03-decision-thread-sweep
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
  shutdown -h now
}
trap finish EXIT

EGRESS=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 https://s3.amazonaws.com 2>/dev/null)
[ "$EGRESS" = "000" ] && { STATUS=NO_EGRESS; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || say "apt update failed (continuing)"
apt-get install -y -qq unzip python3 curl || say "apt install failed (continuing)"
if ! command -v aws >/dev/null 2>&1; then
  curl -sfL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && (cd /tmp && unzip -qo awscli.zip && ./aws/install --update >/dev/null 2>&1); hash -r
fi
command -v aws >/dev/null 2>&1 || { STATUS=NO_AWS_CLI; exit 1; }

{ echo "== uname =="; uname -srm; echo "== cpu =="; lscpu; echo "== mem =="; head -3 /proc/meminfo; } > "$OUT/machine.txt" 2>&1
aws s3 cp "$OUT/machine.txt" "s3://$BUCKET/results/$RUN/machine.txt" --only-show-errors 2>/dev/null
say "specs: $(grep -m1 'Model name' "$OUT/machine.txt")  cpus=$(nproc)"

curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz \
  || { STATUS=NO_JDK; exit 1; }
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 || { STATUS=NO_JDK_UNPACK; exit 1; }
export JAVA_HOME=/opt/jdk PATH=/opt/jdk/bin:$PATH
say "jdk $(java -version 2>&1 | head -1)"

curl -Ls https://sh.jbang.dev -o /tmp/jbang.sh && bash /tmp/jbang.sh app setup >/dev/null 2>&1
export PATH="$HOME/.jbang/bin:$PATH"
command -v jbang >/dev/null 2>&1 || { STATUS=NO_JBANG; exit 1; }
say "jbang $(jbang version 2>&1 | head -1)"

cd /work
aws s3 cp "s3://$BUCKET/payload/DecideLatency-0003.java" ./DecideLatency.java --only-show-errors \
  || { STATUS=NO_PAYLOAD; exit 1; }

# One warm run first so the 1 GB download and the jbang build are not inside any measured arm.
say "priming (download + build)"
jbang -Dvectors.gguf.threads=4 -Dmodels.native.kernels.threads=4 DecideLatency.java > "$OUT/prime.log" 2>&1
say "prime exit $?"
aws s3 cp "$OUT/prime.log" "s3://$BUCKET/results/$RUN/prime.log" --only-show-errors 2>/dev/null

# Both pools together, then each alone, so a win can be attributed rather than just observed.
for t in 32 16 8 4 2; do
  say "sweep both=$t"
  jbang -Dvectors.gguf.threads=$t -Dmodels.native.kernels.threads=$t \
    DecideLatency.java --json "$OUT/both-$t.json" > "$OUT/both-$t.log" 2>&1
  say "  exit $? -> $(grep -m1 'warm: p50' "$OUT/both-$t.log" || echo no-result)"
done
for t in 32 8; do
  say "sweep java-only=$t (native stays 32)"
  jbang -Dvectors.gguf.threads=$t -Dmodels.native.kernels.threads=32 \
    DecideLatency.java --json "$OUT/java-$t.json" > "$OUT/java-$t.log" 2>&1
  say "  exit $? -> $(grep -m1 'warm: p50' "$OUT/java-$t.log" || echo no-result)"
  say "sweep native-only=$t (java stays 32)"
  jbang -Dvectors.gguf.threads=32 -Dmodels.native.kernels.threads=$t \
    DecideLatency.java --json "$OUT/native-$t.json" > "$OUT/native-$t.log" 2>&1
  say "  exit $? -> $(grep -m1 'warm: p50' "$OUT/native-$t.log" || echo no-result)"
done
# The poll budget the native path forces to 0: does restoring it help at high thread counts?
say "sweep both=32 with poll restored"
jbang -Dvectors.gguf.threads=32 -Dmodels.native.kernels.threads=32 -Dvectors.gguf.pollMillis=5 \
  DecideLatency.java --json "$OUT/poll5-32.json" > "$OUT/poll5-32.log" 2>&1
say "  exit $?"

aws s3 cp "$OUT/" "s3://$BUCKET/results/$RUN/" --recursive --only-show-errors 2>/dev/null
COUNT=$(ls "$OUT"/*.json 2>/dev/null | wc -l | tr -d ' ')
say "uploaded $COUNT json results"
[ "$COUNT" -ge 5 ] && STATUS=OK || STATUS=PARTIAL_$COUNT
exit 0
