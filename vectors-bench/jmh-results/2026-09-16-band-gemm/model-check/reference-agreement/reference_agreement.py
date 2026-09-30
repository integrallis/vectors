#!/usr/bin/env python3
"""Pre-registration 3: float32 reference (dequantised Q4_K_M weights) greedy continuations for the
20 model-check prompts, and agreement of the integer and dispatch arms with it."""
import hashlib, json, pathlib, sys, time
sys.path.insert(0, "/opt/ref")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reference_alora_case import patch_with_gguf

BASE, REV = "ibm-granite/granite-4.1-3b", "c0650403e44e78ec0262dab1c90914c65b196c4e"
GGUF = "/opt/ref/granite-4.1-3b-Q4_K_M.gguf"
EV = "/root/band-dispatch-model-check/evidence/20260917T022838Z"
torch.manual_seed(0)
torch.set_num_threads(16)
prompts = json.load(open("/opt/refagree/prompts.json"))
integer = {c["id"]: c for c in json.load(open(f"{EV}/cont-base-arm-integer.json"))["cases"]}
dispatch = {c["id"]: c for c in json.load(open(f"{EV}/cont-base-arm-dispatch.json"))["cases"]}
gguf_sha = hashlib.sha256(open(GGUF, "rb").read()).hexdigest()
assert gguf_sha.startswith("662b0626"), gguf_sha
tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, torch_dtype=torch.float32)
patch = patch_with_gguf(model, pathlib.Path(GGUF))
model.eval()
rows = []
t0 = time.time()
for p in prompts:
    assert p["shaMatches"]
    ids = tok(p["promptText"], add_special_tokens=False, return_tensors="pt")
    n = int(ids["input_ids"].shape[1])
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
        top2 = torch.topk(logits, 2)
        out = model.generate(**ids, max_new_tokens=8, do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
    gen_ids = out[0, n:].tolist()
    if tok.eos_token_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(tok.eos_token_id)]
    ref = tok.decode(gen_ids, skip_special_tokens=True).strip()
    i_out = integer[p["id"]]["output"].strip()
    d_out = dispatch[p["id"]]["output"].strip()
    row = {
        "id": p["id"], "referenceOutput": ref, "integerOutput": i_out, "dispatchOutput": d_out,
        "integerAgrees": ref == i_out, "dispatchAgrees": ref == d_out, "armsDiverge": i_out != d_out,
        "referencePromptTokens": n, "runtimePromptTokens": p["promptTokens"],
        "firstTokenTop2": [[tok.decode([int(i)]), float(v)] for v, i in zip(top2.values, top2.indices)],
        "firstTokenMargin": float(top2.values[0] - top2.values[1]),
    }
    rows.append(row)
    print("%s ref=%r int=%r disp=%r margin=%.3f tokens ref/rt=%d/%d %.0fs" % (
        row["id"], ref, i_out, d_out, row["firstTokenMargin"], n, p["promptTokens"], time.time() - t0), flush=True)
agree_int = sum(r["integerAgrees"] for r in rows)
agree_disp = sum(r["dispatchAgrees"] for r in rows)
div = [r for r in rows if r["armsDiverge"]]
div_band = sum(r["dispatchAgrees"] for r in div)
div_int = sum(r["integerAgrees"] for r in div)
verdict = {
    "agreementInteger": agree_int, "agreementDispatch": agree_disp, "divergentCases": len(div),
    "divergentBandAgrees": div_band, "divergentIntegerAgrees": div_int,
    "a_bandAtLeastInteger": agree_disp >= agree_int, "b_bandAgreesOnAtLeast2Divergent": div_band >= 2,
    "faithful": agree_disp >= agree_int and div_band >= 2,
    "tokenCountMismatches": sum(1 for r in rows if r["referencePromptTokens"] != r["runtimePromptTokens"]),
    "ggufSha256": gguf_sha, "maxRelativeWeightError": patch["maxRelativeError"], "dtype": "float32",
}
json.dump({"verdict": verdict, "cases": rows}, open("/opt/refagree/reference-agreement.json", "w"), indent=2)
print("VERDICT", json.dumps(verdict), flush=True)
