#!/usr/bin/env python3
"""Pre-registration 4: float32 Transformers reference hidden states for the layer probe.

For each prompt id, reads the Java probe output ``DIR/CONDITION-ARM-ID.json`` (any arm: token ids
and prompt text are arm-independent and are cross-checked between the files given), runs
``ibm-granite/granite-4.1-3b`` at the pinned revision with its weights replaced by the dequantised
tensors of the same GGUF, and writes ``DIR/reference-ID.json`` with the last-position vector at the
stages the Java probe emits: ``embedding`` (input of decoder layer 0, i.e. after Granite's
embedding_multiplier), ``layer.L`` (output of decoder layer L), ``final_norm`` (output of
``model.norm``) and ``logits`` (model output, i.e. after the division by logits_scaling).

Stages come from module hooks, not from the ``output_hidden_states`` tuple, whose last entry is
post-norm in current Transformers. The tuple is captured too and cross-checked against the hooks;
the observed convention is recorded under ``hiddenStatesConvention``.

``--local-transfer ARM[,ARM]`` (needs the Java ``fresh`` all-positions dumps): for every layer L,
runs that arm's full-sequence input to layer L through the reference layer L with the reference's
own layer kwargs (mask, rotary embeddings), and compares the result with that arm's layer L output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import time

BASE, REV = "ibm-granite/granite-4.1-3b", "c0650403e44e78ec0262dab1c90914c65b196c4e"


def l2(values) -> float:
    return float(values.double().norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=pathlib.Path)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--probe-files", default="pipeline-cache-integer,pipeline-cache-dispatch",
                        help="comma list of CONDITION-ARM prefixes whose token ids must all agree")
    parser.add_argument("--gguf", required=True,
                        help="GGUF whose dequantised tensors replace the HF weights ('none' only for smoke tests)")
    parser.add_argument("--model-id", default=BASE)
    parser.add_argument("--revision", default=REV)
    parser.add_argument("--gguf-sha256-prefix", default="662b0626")
    parser.add_argument("--reference-module-dir", default="/opt/ref")
    parser.add_argument("--local-transfer", default="",
                        help="comma list of arms with fresh-ARM-ID.f32 dumps, e.g. integer,dispatch")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    sys.path.insert(0, args.reference_module_dir)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision, torch_dtype=torch.float32)
    if args.gguf == "none":
        gguf_sha, patch = None, {"replaced": 0, "maxRelativeError": None}
    else:
        from reference_alora_case import patch_with_gguf
        digest = hashlib.sha256()
        with open(args.gguf, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        gguf_sha = digest.hexdigest()
        if not gguf_sha.startswith(args.gguf_sha256_prefix):
            raise SystemExit(f"GGUF sha256 {gguf_sha} does not start with {args.gguf_sha256_prefix}")
        patch = patch_with_gguf(model, pathlib.Path(args.gguf))
    model.eval()
    inner = model.model
    layers = inner.layers
    n_layers = len(layers)

    captured: dict = {}

    def pre_hook(index):
        def hook(module, hook_args, hook_kwargs):
            hidden = hook_args[0] if hook_args else hook_kwargs["hidden_states"]
            captured[("in", index)] = hidden.detach().clone()
            kwargs = {k: v for k, v in hook_kwargs.items() if k != "hidden_states"}
            captured[("kwargs", index)] = kwargs
        return hook

    def out_hook(key):
        def hook(module, hook_args, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[key] = value.detach().clone()
        return hook

    handles = []
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(pre_hook(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(out_hook(("out", index))))
        handles.append(layer.self_attn.register_forward_hook(out_hook(("attn", index))))
        handles.append(layer.mlp.register_forward_hook(out_hook(("mlp", index))))
    handles.append(inner.norm.register_forward_hook(out_hook(("norm",))))

    local_arms = [a for a in args.local_transfer.split(",") if a]
    for case_id in args.ids.split(","):
        started = time.time()
        probes = {}
        for prefix in args.probe_files.split(","):
            path = args.dir / f"{prefix}-{case_id}.json"
            if path.exists():
                probes[prefix] = json.loads(path.read_text())
        if not probes:
            raise SystemExit(f"no Java probe output for {case_id} in {args.dir}")
        first = next(iter(probes.values()))
        for name, probe in probes.items():
            if probe["tokenIds"] != first["tokenIds"] or probe["promptTextSha256"] != first["promptTextSha256"]:
                raise SystemExit(f"{case_id}: token ids / prompt SHA differ between probe files ({name})")
        text = first["promptText"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != first["promptTextSha256"]:
            raise SystemExit(f"{case_id}: prompt text does not match its SHA-256")
        java_ids = first["tokenIds"]
        hf_ids = tok(text, add_special_tokens=False)["input_ids"]
        assert len(hf_ids) == len(java_ids), (case_id, len(hf_ids), len(java_ids))
        ids_equal = list(hf_ids) == list(java_ids)
        id_mismatch_positions = [i for i, (a, b) in enumerate(zip(hf_ids, java_ids)) if a != b]

        captured.clear()
        input_ids = torch.tensor([java_ids], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        logits = output.logits[0, -1].float()
        last = lambda t: t[0, -1].float()

        stages = ["embedding"] + [f"layer.{i}" for i in range(n_layers)] + ["final_norm", "logits"]
        vectors = [last(captured[("in", 0)])]
        vectors += [last(captured[("out", i)]) for i in range(n_layers)]
        vectors += [last(captured[("norm",)]), logits]

        # Hook consistency (hard gate): layer i's captured output must be exactly layer i+1's input.
        chain = max(float((captured[("out", i)] - captured[("in", i + 1)]).abs().max())
                    for i in range(n_layers - 1))
        if chain != 0.0:
            raise SystemExit(f"{case_id}: layer output hooks do not chain into the next layer input ({chain})")
        # output_hidden_states convention (recorded; a mismatch is reported, the hooks stay primary).
        hs = output.hidden_states
        convention = {"length": len(hs), "numLayers": n_layers, "hookChainMaxAbs": chain}
        if len(hs) == n_layers + 1:
            convention["hs0_vs_layer0_input_maxabs"] = float((hs[0] - captured[("in", 0)]).abs().max())
            convention["hs_i_vs_layer_i-1_output_maxabs_max_over_1..n-1"] = max(
                float((hs[i] - captured[("out", i - 1)]).abs().max()) for i in range(1, n_layers))
            convention["hs_last_vs_last_layer_output_maxabs"] = float(
                (hs[-1] - captured[("out", n_layers - 1)]).abs().max())
            convention["hs_last_vs_final_norm_maxabs"] = float((hs[-1] - captured[("norm",)]).abs().max())
            convention["hs_last_is"] = (
                "final_norm" if convention["hs_last_vs_final_norm_maxabs"] == 0.0
                else "last_layer_output" if convention["hs_last_vs_last_layer_output_maxabs"] == 0.0
                else "neither")
            convention["tupleAgreesWithHooks"] = (
                convention["hs0_vs_layer0_input_maxabs"] == 0.0
                and convention["hs_i_vs_layer_i-1_output_maxabs_max_over_1..n-1"] == 0.0)
        else:
            convention["tupleAgreesWithHooks"] = False
        if not convention["tupleAgreesWithHooks"]:
            print(f"WARNING {case_id}: output_hidden_states does not match hook stages: {convention}",
                  flush=True)
        del hs

        # Per-layer sublayer update sizes at the last position (reference only).
        r = float(model.config.residual_multiplier)
        sublayer = []
        for i in range(n_layers):
            sublayer.append({
                "layer": i,
                "inputNorm": l2(last(captured[("in", i)])),
                "attentionUpdateNorm": l2(last(captured[("attn", i)])) * r,
                "mlpUpdateNorm": l2(last(captured[("mlp", i)])) * r,
            })

        top = torch.topk(logits, 5)
        report = {
            "id": case_id,
            "model": args.model_id, "revision": args.revision, "dtype": "float32",
            "ggufSha256": gguf_sha, "patchedTensors": patch["replaced"],
            "maxRelativeWeightChangeVsHf": patch["maxRelativeError"],
            "promptTextSha256": first["promptTextSha256"],
            "promptTokens": len(java_ids), "tokenIds": java_ids,
            "hfTokenizerIdsEqualJava": ids_equal,
            "hfTokenizerIdMismatchPositions": id_mismatch_positions[:50],
            "ranOn": "Java token ids",
            "config": {
                "embedding_multiplier": getattr(model.config, "embedding_multiplier", None),
                "residual_multiplier": getattr(model.config, "residual_multiplier", None),
                "logits_scaling": getattr(model.config, "logits_scaling", None),
                "attention_multiplier": getattr(model.config, "attention_multiplier", None),
                "rms_norm_eps": getattr(model.config, "rms_norm_eps", None),
                "attn_implementation": getattr(model.config, "_attn_implementation", None),
                "architectures": getattr(model.config, "architectures", None),
            },
            "stageProvenance": {
                "embedding": "forward pre-hook: input of model.layers[0] (after embedding_multiplier)",
                "layer.L": "forward hook: output of model.layers[L]",
                "final_norm": "forward hook: output of model.norm",
                "logits": "model output logits (after /logits_scaling)",
            },
            "hiddenStatesConvention": convention,
            "sublayerUpdates": sublayer,
            "top5": [{"token": int(i), "text": tok.decode([int(i)]), "logit": float(v)}
                     for v, i in zip(top.values, top.indices)],
            "stages": stages,
            "vectors": [v.tolist() for v in vectors],
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "machine": platform.machine()},
        }

        if local_arms:
            report["localTransfer"] = local_transfer(
                args.dir, case_id, local_arms, layers, captured, n_layers, len(java_ids), np, torch, r)

        report["seconds"] = round(time.time() - started, 1)
        out = args.dir / f"reference-{case_id}.json"
        out.write_text(json.dumps(report))
        print(f"reference {case_id} tokens={len(java_ids)} idsEqual={ids_equal} "
              f"hs_last_is={convention['hs_last_is']} top1={report['top5'][0]['text']!r} "
              f"{report['seconds']}s -> {out}", flush=True)

    for handle in handles:
        handle.remove()


def local_transfer(directory, case_id, arms, layers, captured, n_layers, tokens, np, torch, r):
    """One-layer transfer: Java arm's input to layer L through reference layer L vs Java's output."""
    result = {"method": "reference layer L applied to the arm's full-sequence layer L input, "
                        "reference layer kwargs reused; compared at the last position. "
                        "attention/mlp update norms are the reference sublayers' updates on that input"}
    # Snapshot first: the module hooks overwrite `captured` on every replay call below.
    kwargs = {i: captured[("kwargs", i)] for i in range(n_layers)}
    ref_in = {i: captured[("in", i)] for i in range(min(2, n_layers))}
    ref_out = {i: captured[("out", i)] for i in range(min(2, n_layers))}
    # Self-check: replaying the reference's own layer input with the captured kwargs must reproduce
    # its own layer output, or the replay itself is unfaithful and the local numbers mean nothing.
    self_check = []
    with torch.no_grad():
        for i in ref_in:
            replay = layers[i](ref_in[i], **kwargs[i])
            replay = replay[0] if isinstance(replay, (tuple, list)) else replay
            self_check.append(float((replay - ref_out[i]).abs().max()))
    result["selfReplayMaxAbs"] = self_check
    for arm in arms:
        meta = json.loads((directory / f"fresh-{arm}-{case_id}.json").read_text())
        shape = tuple(meta["allPositionsShape"])
        if shape[0] != n_layers + 1 or shape[1] != tokens:
            raise SystemExit(f"{case_id} {arm}: dump shape {shape} does not match {n_layers + 1}x{tokens}")
        data = np.memmap(directory / meta["allPositionsFile"], dtype="<f4", mode="r", shape=shape)
        rows = []
        with torch.no_grad():
            for i in range(n_layers):
                java_in = torch.from_numpy(np.array(data[i])).unsqueeze(0)
                java_out = torch.from_numpy(np.array(data[i + 1, -1])).double()
                out = layers[i](java_in, **kwargs[i])
                out = out[0] if isinstance(out, (tuple, list)) else out
                ref_last = out[0, -1].double()
                in_last = java_in[0, -1].double()
                diff = float((java_out - ref_last).norm())
                rows.append({
                    "layer": i,
                    "localRelErr": diff / (float(ref_last.norm()) or 1.0),
                    "localErrOverUpdate": diff / (float((ref_last - in_last).norm()) or 1.0),
                    "attentionUpdateNorm": float(captured[("attn", i)][0, -1].double().norm()) * r,
                    "mlpUpdateNorm": float(captured[("mlp", i)][0, -1].double().norm()) * r,
                })
        result[arm] = rows
        del data
    return result


if __name__ == "__main__":
    main()
