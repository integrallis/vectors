#!/usr/bin/env python3
"""Synthetic tests for compare_layers.py alignment gate and pre-registered rule."""
import json
import math
import pathlib
import random
import tempfile
import unittest

import compare_layers as cl

DIM = 16
LAYERS = 4
STAGES = ["embedding"] + [f"layer.{i}" for i in range(LAYERS)] + ["final_norm", "logits"]


def vector(seed, scale=1.0, size=DIM):
    rng = random.Random(seed)
    return [rng.gauss(0, scale) for _ in range(size)]


def perturb(v, relative, seed):
    """Adds noise of exactly `relative` * ||v|| in L2."""
    noise = vector(seed, size=len(v))
    factor = relative * cl.norm(v) / cl.norm(noise)
    return [x + factor * n for x, n in zip(v, noise)]


def reference_doc(case_id="c1"):
    vectors = [vector(100 + i, size=32 if s == "logits" else DIM) for i, s in enumerate(STAGES)]
    return {"id": case_id, "stages": list(STAGES), "vectors": vectors, "tokenIds": [1, 2, 3],
            "promptTextSha256": "abc", "top5": [{"text": "x"}]}


def arm_doc(reference, errors, seed, identical_embedding=True):
    vectors = []
    for i, stage in enumerate(STAGES):
        if stage == "embedding" and identical_embedding:
            vectors.append(list(reference["vectors"][i]))
        else:
            vectors.append(perturb(reference["vectors"][i], errors[stage], seed * 1000 + i))
    return {"id": reference["id"], "stages": list(STAGES), "vectors": vectors,
            "tokenIds": list(reference["tokenIds"]), "promptTextSha256": reference["promptTextSha256"],
            "promptTokens": 3, "prefillStartPosition": 0, "prefillTokens": 3, "prefillBatchCapacity": 512,
            "vectorsKernel": "integer" if seed == 1 else "dispatch(band-at-batch>=...)", "reproducesRecordedFirstToken": True, "argmaxText": "x",
            "finalNormRecomputedMaxAbsDiff": 0.0}


def growing(base):
    errors = {"embedding": 0.0}
    for i in range(LAYERS):
        errors[f"layer.{i}"] = base * (i + 1)
    errors["final_norm"] = base * (LAYERS + 1)
    errors["logits"] = base * (LAYERS + 2)
    return errors


class RelTest(unittest.TestCase):
    def test_relative_error_is_exact(self):
        v = vector(1)
        self.assertAlmostEqual(cl.rel(perturb(v, 0.1, 2), v, v), 0.1, places=12)

    def test_length_mismatch_is_misaligned(self):
        with self.assertRaises(cl.Misaligned):
            cl.rel([1.0], [1.0, 2.0], [1.0])


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.ref = reference_doc()
        self.integer = arm_doc(self.ref, growing(1e-3), seed=1)
        self.band = arm_doc(self.ref, growing(1e-3), seed=2)

    def test_aligned_files_pass(self):
        self.assertEqual(cl.check_alignment(self.integer, self.band, self.ref), [])

    def test_off_by_one_layer_is_detected(self):
        shifted = dict(self.integer)
        vectors = list(self.integer["vectors"])
        # Java layer.0 slot holds what is really layer.1 (observer index shifted by one).
        vectors[1] = list(self.ref["vectors"][2])
        shifted["vectors"] = vectors
        problems = cl.check_alignment(shifted, self.band, self.ref)
        self.assertTrue(any("layer.0" in p for p in problems), problems)

    def test_embedding_mismatch_is_detected(self):
        bad = arm_doc(self.ref, dict(growing(1e-3), embedding=1e-2), seed=3, identical_embedding=False)
        problems = cl.check_alignment(bad, self.band, self.ref)
        self.assertTrue(any("e[embedding]" in p for p in problems), problems)

    def test_stage_list_mismatch(self):
        other = dict(self.band, stages=STAGES[:-1], vectors=self.band["vectors"][:-1])
        self.assertIn("stage lists differ", cl.check_alignment(self.integer, other, self.ref))

    def test_vector_length_mismatch(self):
        other = dict(self.band)
        other["vectors"] = [list(v) for v in self.band["vectors"]]
        other["vectors"][3] = other["vectors"][3][:-1]
        problems = cl.check_alignment(self.integer, other, self.ref)
        self.assertTrue(any("vector lengths" in p for p in problems), problems)

    def test_token_ids_mismatch(self):
        other = dict(self.band, tokenIds=[1, 2, 4])
        self.assertIn("token ids differ", cl.check_alignment(self.integer, other, self.ref))

    def test_compare_exits_2_on_misalignment_and_0_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "reference-c1.json").write_text(json.dumps(self.ref))
            (d / "pipeline-cache-integer-c1.json").write_text(json.dumps(self.integer))
            (d / "pipeline-cache-dispatch-c1.json").write_text(json.dumps(self.band))
            self.assertEqual(cl.main(["--dir", tmp, "--ids", "c1", "--json", str(d / "out.json")]), 0)
            self.assertIn("faithful", json.loads((d / "out.json").read_text()))
            wrong_kernel = dict(self.band, vectorsKernel="integer")
            (d / "pipeline-cache-dispatch-c1.json").write_text(json.dumps(wrong_kernel))
            self.assertEqual(cl.main(["--dir", tmp, "--ids", "c1"]), 2)
            bad = dict(self.band, tokenIds=[9, 9, 9])
            (d / "pipeline-cache-dispatch-c1.json").write_text(json.dumps(bad))
            self.assertEqual(cl.main(["--dir", tmp, "--ids", "c1"]), 2)
            self.assertEqual(cl.main(["--dir", tmp, "--ids", "missing"]), 2)


class RuleTest(unittest.TestCase):
    def rows(self, e_int, e_band):
        return [{"stage": s, "e_int": e_int[s], "e_band": e_band[s], "e_ib": 0.0} for s in STAGES]

    def test_equal_errors_are_faithful(self):
        verdict = cl.apply_rule(self.rows(growing(1e-3), growing(1e-3)))
        self.assertTrue(verdict["faithful"])
        self.assertIsNone(verdict["firstViolation"])

    def test_ratio_boundary_is_inclusive(self):
        e_int = growing(1e-3)
        e_band = {k: v * 1.25 for k, v in e_int.items()}
        e_band["logits"] = e_int["logits"]
        self.assertTrue(cl.apply_rule(self.rows(e_int, e_band))["faithful"])

    def test_first_violation_is_located(self):
        e_int = growing(1e-3)
        e_band = dict(e_int)
        e_band["layer.2"] = e_int["layer.2"] * 1.3
        e_band["layer.3"] = e_int["layer.3"] * 5
        verdict = cl.apply_rule(self.rows(e_int, e_band))
        self.assertFalse(verdict["faithful"])
        self.assertEqual(verdict["firstViolation"]["stage"], "layer.2")
        self.assertEqual(verdict["firstViolation"]["nextStage"]["stage"], "layer.3")
        self.assertAlmostEqual(verdict["firstViolation"]["ratio"], 1.3)
        self.assertFalse(verdict["firstViolation"]["bothBelowRoundingLevel"])

    def test_logits_worse_fails_even_when_ratio_holds(self):
        e_int = growing(1e-3)
        e_band = dict(e_int, logits=e_int["logits"] * 1.1)
        verdict = cl.apply_rule(self.rows(e_int, e_band))
        self.assertTrue(verdict["ratioHoldsEverywhere"])
        self.assertFalse(verdict["logitsBandNotWorse"])
        self.assertFalse(verdict["faithful"])

    def test_zero_integer_error_with_nonzero_band_is_violation(self):
        e_int = growing(1e-3)
        e_band = dict(e_int, embedding=1e-9)
        verdict = cl.apply_rule(self.rows(e_int, e_band))
        self.assertEqual(verdict["firstViolation"]["stage"], "embedding")
        self.assertTrue(math.isinf(verdict["firstViolation"]["ratio"]))
        self.assertTrue(verdict["firstViolation"]["bothBelowRoundingLevel"])

    def test_errors_from_documents(self):
        ref = reference_doc()
        integer = arm_doc(ref, growing(1e-3), seed=1)
        band = arm_doc(ref, dict(growing(1e-3), **{"layer.1": 5e-3}), seed=2)
        rows = cl.errors(integer, band, ref)
        by_stage = {r["stage"]: r for r in rows}
        self.assertAlmostEqual(by_stage["layer.1"]["e_band"], 5e-3, places=12)
        self.assertAlmostEqual(by_stage["layer.1"]["e_int"], 2e-3, places=12)
        self.assertEqual(cl.apply_rule(rows)["firstViolation"]["stage"], "layer.1")


if __name__ == "__main__":
    unittest.main()
