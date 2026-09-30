#!/usr/bin/env python3
"""Pre-registration 4: per-layer comparison of the integer and band (dispatch) arms against the
float reference, alignment gate, and the pre-registered decision rule.

    compare_layers.py --dir D --ids a,b,c [--condition pipeline-cache] [--json out.json]

Reads D/CONDITION-integer-ID.json, D/CONDITION-dispatch-ID.json and D/reference-ID.json.
Exit status: 0 when every prompt passes the alignment gate (whatever the verdict), 2 on any
shape / stage / token / index misalignment. Pure Python so the rule logic is testable anywhere.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

RATIO = 1.25
EMBEDDING_MAX = 1e-4
LAYER0_MAX = 0.05
ROUNDING_LEVEL = 1e-5


class Misaligned(Exception):
    pass


def norm(a) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def diff_norm(a, b) -> float:
    return math.sqrt(math.fsum((x - y) * (x - y) for x, y in zip(a, b)))


def rel(a, b, ref) -> float:
    """||a - b|| / ||ref||."""
    if len(a) != len(b) or len(a) != len(ref):
        raise Misaligned(f"vector lengths differ: {len(a)} {len(b)} {len(ref)}")
    denominator = norm(ref)
    return diff_norm(a, b) / denominator if denominator else (0.0 if diff_norm(a, b) == 0 else math.inf)


def check_alignment(integer: dict, band: dict, reference: dict) -> list[str]:
    """Returns the list of alignment problems; empty means the three files are comparable."""
    problems = []
    if not (integer["stages"] == band["stages"] == reference["stages"]):
        problems.append("stage lists differ")
        return problems
    stages = integer["stages"]
    for name, doc in (("integer", integer), ("band", band), ("reference", reference)):
        if len(doc["vectors"]) != len(stages):
            problems.append(f"{name}: {len(doc['vectors'])} vectors for {len(stages)} stages")
    if problems:
        return problems
    for index, stage in enumerate(stages):
        lengths = {len(integer["vectors"][index]), len(band["vectors"][index]), len(reference["vectors"][index])}
        if len(lengths) != 1:
            problems.append(f"{stage}: vector lengths {sorted(lengths)}")
    if not (integer["tokenIds"] == band["tokenIds"] == reference["tokenIds"]):
        problems.append("token ids differ")
    if not (integer["promptTextSha256"] == band["promptTextSha256"] == reference["promptTextSha256"]):
        problems.append("prompt SHA-256 differs")
    if integer.get("id") != band.get("id") or integer.get("id") != reference.get("id"):
        problems.append("prompt ids differ")
    for required in ("embedding", "layer.0", "layer.1", "final_norm", "logits"):
        if required not in stages:
            problems.append(f"missing stage {required}")
    if problems:
        return problems
    at = {stage: i for i, stage in enumerate(stages)}
    ref = reference["vectors"]
    for name, doc in (("integer", integer), ("band", band)):
        vec = doc["vectors"]
        e_embedding = rel(vec[at["embedding"]], ref[at["embedding"]], ref[at["embedding"]])
        if e_embedding > EMBEDDING_MAX:
            problems.append(f"{name}: e[embedding]={e_embedding:.3g} > {EMBEDDING_MAX}")
        aligned = rel(vec[at["layer.0"]], ref[at["layer.0"]], ref[at["layer.0"]])
        if aligned > LAYER0_MAX:
            problems.append(f"{name}: e[layer.0]={aligned:.3g} > {LAYER0_MAX}")
        for neighbour in ("embedding", "layer.1"):
            shifted = rel(vec[at["layer.0"]], ref[at[neighbour]], ref[at["layer.0"]])
            if not aligned < shifted:
                problems.append(
                    f"{name}: layer.0 is not closer to reference layer.0 ({aligned:.3g}) "
                    f"than to reference {neighbour} ({shifted:.3g})")
    return problems


def errors(integer: dict, band: dict, reference: dict) -> list[dict]:
    rows = []
    for index, stage in enumerate(reference["stages"]):
        ref = reference["vectors"][index]
        vi = integer["vectors"][index]
        vb = band["vectors"][index]
        rows.append({
            "stage": stage,
            "e_int": rel(vi, ref, ref),
            "e_band": rel(vb, ref, ref),
            "e_ib": rel(vi, vb, ref),
        })
    return rows


def apply_rule(rows: list[dict]) -> dict:
    """Pre-registered rule: faithful iff e_band <= 1.25 e_int at every stage and
    e_band[logits] <= e_int[logits]; L* is the first stage breaking the ratio."""
    first = None
    for row in rows:
        if row["e_band"] > RATIO * row["e_int"]:
            first = row
            break
    logits = next(r for r in rows if r["stage"] == "logits")
    logits_ok = logits["e_band"] <= logits["e_int"]
    verdict = {
        "faithful": first is None and logits_ok,
        "ratioHoldsEverywhere": first is None,
        "logitsBandNotWorse": logits_ok,
        "firstViolation": None,
    }
    if first is not None:
        index = rows.index(first)
        following = rows[index + 1] if index + 1 < len(rows) else None
        verdict["firstViolation"] = {
            "stage": first["stage"],
            "e_int": first["e_int"],
            "e_band": first["e_band"],
            "ratio": first["e_band"] / first["e_int"] if first["e_int"] else math.inf,
            # Pre-registered context, not a relaxation: both errors at float32 rounding level.
            "bothBelowRoundingLevel": first["e_int"] < ROUNDING_LEVEL and first["e_band"] < ROUNDING_LEVEL,
            "nextStage": None if following is None else {
                "stage": following["stage"],
                "ratio": following["e_band"] / following["e_int"] if following["e_int"] else math.inf,
            },
        }
    return verdict


def ratio_text(row: dict) -> str:
    if row["e_int"] == 0:
        return "inf" if row["e_band"] else "-"
    return f"{row['e_band'] / row['e_int']:.3f}"


def compare_prompt(directory: pathlib.Path, condition: str, case_id: str) -> dict:
    load = lambda name: json.loads((directory / name).read_text())
    integer = load(f"{condition}-integer-{case_id}.json")
    band = load(f"{condition}-dispatch-{case_id}.json")
    reference = load(f"reference-{case_id}.json")
    problems = check_alignment(integer, band, reference)
    # A kernel toggle that silently did nothing would make the two arms the same arm.
    if integer.get("vectorsKernel") != "integer":
        problems.append(f"integer file ran kernel {integer.get('vectorsKernel')!r}")
    if not str(band.get("vectorsKernel", "")).startswith("dispatch"):
        problems.append(f"dispatch file ran kernel {band.get('vectorsKernel')!r}")
    if problems:
        raise Misaligned(f"{case_id}: " + "; ".join(problems))
    rows = errors(integer, band, reference)
    return {
        "id": case_id,
        "condition": condition,
        "promptTokens": integer["promptTokens"],
        "prefill": {arm: {"start": doc["prefillStartPosition"], "tokens": doc["prefillTokens"],
                          "batchCapacity": doc["prefillBatchCapacity"],
                          "lastPositionChunkSize": doc.get("lastPositionChunkSize")}
                    for arm, doc in (("integer", integer), ("dispatch", band))},
        "kernels": {"integer": integer["vectorsKernel"], "dispatch": band["vectorsKernel"]},
        "reproducesRecordedFirstToken": {
            "integer": integer["reproducesRecordedFirstToken"],
            "dispatch": band["reproducesRecordedFirstToken"]},
        "argmax": {"integer": integer["argmaxText"], "dispatch": band["argmaxText"],
                   "reference": reference["top5"][0]["text"]},
        "observerLogitsBitIdentical": {
            "integer": integer.get("observerLogitsBitIdentical"),
            "dispatch": band.get("observerLogitsBitIdentical")},
        "finalNormRecomputedMaxAbsDiff": {
            "integer": integer["finalNormRecomputedMaxAbsDiff"],
            "dispatch": band["finalNormRecomputedMaxAbsDiff"]},
        "hfTokenizerIdsEqualJava": reference.get("hfTokenizerIdsEqualJava"),
        "hiddenStatesConvention": reference.get("hiddenStatesConvention"),
        "rows": rows,
        "verdict": apply_rule(rows),
        "localTransfer": reference.get("localTransfer"),
    }


def render(result: dict) -> str:
    lines = [f"### {result['id']} ({result['condition']}, {result['promptTokens']} tokens)", ""]
    lines.append(f"- prefill: {json.dumps(result['prefill'])}")
    lines.append(f"- argmax: {json.dumps(result['argmax'])}; reproduces recorded first token: "
                 f"{json.dumps(result['reproducesRecordedFirstToken'])}")
    if any(v is not None for v in result["observerLogitsBitIdentical"].values()):
        lines.append(f"- observer logits bit-identical (fresh control): "
                     f"{json.dumps(result['observerLogitsBitIdentical'])}")
    lines.append(f"- final_norm recomputed max|diff|: {json.dumps(result['finalNormRecomputedMaxAbsDiff'])}; "
                 f"HF tokenizer ids equal Java: {result['hfTokenizerIdsEqualJava']}")
    lines += ["", "| stage | e_int | e_band | e_ib | e_band/e_int | |", "|---|---:|---:|---:|---:|---|"]
    for row in result["rows"]:
        flag = "**> 1.25**" if row["e_band"] > RATIO * row["e_int"] else ""
        lines.append(f"| {row['stage']} | {row['e_int']:.3e} | {row['e_band']:.3e} | {row['e_ib']:.3e} "
                     f"| {ratio_text(row)} | {flag} |")
    verdict = result["verdict"]
    lines.append("")
    lines.append(f"**Rule:** faithful={verdict['faithful']} (ratio everywhere={verdict['ratioHoldsEverywhere']}, "
                 f"logits band<=int={verdict['logitsBandNotWorse']})")
    if verdict["firstViolation"]:
        v = verdict["firstViolation"]
        lines.append(f"**L\\* = {v['stage']}** ratio {v['ratio']:.3f} (e_int {v['e_int']:.3e}, e_band {v['e_band']:.3e}); "
                     f"both below {ROUNDING_LEVEL:g}: {v['bothBelowRoundingLevel']}; next stage {json.dumps(v['nextStage'])}")
    local = result.get("localTransfer")
    if local:
        lines += ["", f"Local one-layer transfer (supplementary; self-replay max|diff| {local.get('selfReplayMaxAbs')})", "",
                  "| layer | local int | local band | band/int | int err/update | band err/update |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for li, lb in zip(local.get("integer", []), local.get("dispatch", [])):
            r = lb["localRelErr"] / li["localRelErr"] if li["localRelErr"] else math.inf
            lines.append(f"| {li['layer']} | {li['localRelErr']:.3e} | {lb['localRelErr']:.3e} | {r:.3f} "
                         f"| {li['localErrOverUpdate']:.3e} | {lb['localErrOverUpdate']:.3e} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=pathlib.Path)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--condition", default="pipeline-cache")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args(argv)
    results = []
    try:
        for case_id in args.ids.split(","):
            results.append(compare_prompt(args.dir, args.condition, case_id))
    except (Misaligned, KeyError, FileNotFoundError) as failure:
        print(f"MISALIGNED: {failure}", file=sys.stderr)
        return 2
    for result in results:
        print(render(result))
        print()
    overall = all(r["verdict"]["faithful"] for r in results)
    print(f"## Overall ({args.condition}): band faithful on {sum(r['verdict']['faithful'] for r in results)}"
          f"/{len(results)} prompts -> {'FAITHFUL' if overall else 'NOT FAITHFUL under the pre-registered rule'}")
    if args.json:
        args.json.write_text(json.dumps({"condition": args.condition, "faithful": overall, "prompts": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
