#!/usr/bin/env python3
"""Reads one model-check evidence directory and prints both pre-registered results."""
import glob, json, os, re, statistics, sys

ev = sys.argv[1]
ROUTE = re.compile(r"vectors-gguf-batched-matmul-routing mode=(\S+)((?: \S+ integer=\d+/\d+ band=\d+/\d+)+)")


def routing(text):
    m = ROUTE.findall(text)
    if not m:
        return None
    mode, rest = m[-1]
    formats = {f: (int(ic), int(ir), int(bc), int(br)) for f, ic, ir, bc, br in
               re.findall(r"(\S+) integer=(\d+)/(\d+) band=(\d+)/(\d+)", rest)}
    return mode, formats


problems = []


def check_routing(label, arm, route):
    if route is None:
        problems.append(f"{label}: no routing report (switch not observable)")
        return "missing"
    mode, formats = route
    band_rows = sum(v[3] for v in formats.values())
    int_rows = sum(v[1] for v in formats.values())
    if mode != arm:
        problems.append(f"{label}: reported mode {mode} != requested {arm}")
    if arm == "integer" and band_rows:
        problems.append(f"{label}: integer arm ran band rows={band_rows}")
    if arm == "dispatch" and band_rows == 0:
        problems.append(f"{label}: dispatch arm never reached the band kernel (toggle did nothing)")
    return " ".join(f"{f}:int {v[0]}/{v[1]} band {v[2]}/{v[3]}" for f, v in formats.items()) + f" (mode={mode}, int rows={int_rows}, band rows={band_rows})"


print("# Band/integer dispatch — model-level check (pre-registration 2)\n")
print(open(os.path.join(ev, "host.txt")).readline().strip() + "\n")

# 1. prefill
rows, tps = [], {"integer": [], "dispatch": []}
for path in sorted(glob.glob(os.path.join(ev, "prefill-*-*.log"))):
    arm, rep = re.match(r".*prefill-(integer|dispatch)-(\d+)\.log", path).groups()
    text = open(path).read()
    m = re.search(r"prefill profile: prompt=(\d+) warmups=(\d+) prefill=([\d.]+) tok/s checksum=(\S+)", text)
    if not m:
        problems.append(f"prefill {arm} rep {rep}: no result line")
        continue
    tps[arm].append(float(m.group(3)))
    rows.append((int(rep), arm, int(m.group(1)), float(m.group(3)), m.group(4), check_routing(f"prefill {arm} {rep}", arm, routing(text))))
print("## 1. Prefill tok/s (context 2048)\n")
print("| rep | arm | prompt tokens | tok/s | logit checksum | routing |\n|---:|---|---:|---:|---|---|")
for r in sorted(rows):
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]} | {r[5]} |")
if tps["integer"] and tps["dispatch"]:
    mi, md = statistics.median(tps["integer"]), statistics.median(tps["dispatch"])
    gain = md / mi - 1
    verdict = "PASS" if gain >= 0.10 else "FAIL"
    print(f"\n**PREFILL: median integer {mi:.2f} tok/s, dispatch {md:.2f} tok/s, change {gain*100:+.1f}% "
          f"(min/max integer {min(tps['integer']):.2f}/{max(tps['integer']):.2f}, dispatch "
          f"{min(tps['dispatch']):.2f}/{max(tps['dispatch']):.2f}) -> {verdict} (gate >= +10%)**\n")
else:
    print("\n**PREFILL: INCOMPLETE**\n")

# 2. continuations
print("## 2. Greedy continuations, first 20 squad-v2-dev cases, max 64 tokens\n")
for variant, gating in (("base-arm", True), ("open", False)):
    try:
        a = json.load(open(os.path.join(ev, f"cont-{variant}-integer.json")))
        b = json.load(open(os.path.join(ev, f"cont-{variant}-dispatch.json")))
    except OSError:
        print(f"**{variant}: INCOMPLETE**\n")
        problems.append(f"continuations {variant}: missing output")
        continue
    for rep, arm in ((a, "integer"), (b, "dispatch")):
        m = ROUTE.search(rep["vectorsRouting"])
        check_routing(f"continuations {variant} {arm}", arm, routing(rep["vectorsRouting"]))
    same, lines = 0, []
    for x, y in zip(a["cases"], b["cases"]):
        if x["id"] != y["id"] or x["promptTextSha256"] != y["promptTextSha256"]:
            problems.append(f"{variant}: case/prompt mismatch at {x['id']}")
        identical = x["fragments"] == y["fragments"]
        same += identical
        if not identical:
            k = next((i for i, (p, q) in enumerate(zip(x["fragments"], y["fragments"])) if p != q),
                     min(len(x["fragments"]), len(y["fragments"])))
            lines.append(f"  - {x['id']}: first divergence at token {k}; integer={json.dumps(x['output'])} dispatch={json.dumps(y['output'])}")
    n = len(a["cases"])
    toks = [c["completionTokens"] for c in a["cases"]]
    stops = sorted({c["stopReason"] for c in a["cases"]})
    label = "gate" if gating else "supplementary, not gating"
    verdict = ("PASS" if same >= 19 else "FAIL") if gating else "reported"
    print(f"**CONTINUATIONS {variant} ({label}): identical {same}/{n} -> {verdict}"
          f"{' (gate >= 19/20)' if gating else ''}**  "
          f"integer-arm completion tokens mean {statistics.mean(toks):.1f} (min {min(toks)}, max {max(toks)}), stop reasons {stops}\n")
    if gating and max(toks) < 64 and statistics.mean(toks) < 8:
        print("  note: base-arm outputs are short (one-word instruction), so identity is tested over few tokens; see the open variant.\n")
    for line in lines:
        print(line)
    print()

# prompt parity with the Models runner (optional)
dump = os.path.join(ev, "runner-prompts")
if os.path.isdir(dump):
    base = json.load(open(os.path.join(ev, "cont-base-arm-integer.json")))["cases"]
    match = sum(json.load(open(os.path.join(dump, f"{i}.json")))["textSha256"] == c["promptTextSha256"]
                for i, c in enumerate(base) if os.path.exists(os.path.join(dump, f"{i}.json")))
    print(f"**PROMPT PARITY with activated-answerability --arm base: {match}/{len(base)} byte-identical prompts**\n")
else:
    print("PROMPT PARITY with the Models runner: SKIPPED (ADAPTER_DIR not set)\n")

print("## Observability problems\n")
print("\n".join(f"- {p}" for p in problems) if problems else "- none")
