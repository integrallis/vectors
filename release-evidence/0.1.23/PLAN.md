# Vectors 0.1.23 release qualification

Scope: ship the merged Q6_K scalar reduction-order correction (PR #76), preserving the existing
30-module Apache CPU publication contract. No experimental GEMM or dequantization path is enabled.
Baseline: `27344d4957a027de6057ff9c16d209369e3100c3` (main, 2026-09-25).

## Bounded infrastructure preflight

- Provider: Hetzner; one host at a time; Ubuntu 24.04.
- x86 host: ccx33, nbg1, 8 dedicated vCPU / 32 GB; live USD 0.2612/hour.
- ARM host: cax31, nbg1, 8 vCPU / 16 GB; live USD 0.0400/hour.
- Billing: allow one full hourly increment per allocation and primary IPv4 overhead.
- Total spending ceiling: USD 3.00; no extensions without revising this plan.
- Creation window: 2026-09-25 19:15–21:15 UTC.
- Hard deletion deadline: 2026-09-25 22:15 UTC, including failed/incomplete runs.
- Stop on correctness failure, investigate from copied evidence, do not retry unchanged to green.
- SSH only from operator IPv4 /32; use verified existing public SSH key; no provider secrets on host.
- Preserve logs, XML test results, staged artifact checksums and runtime/CPU details locally before
  deleting the exact owned instance, then check its associated firewall and primary IPs.
- Local raw evidence: `projects/benchmark-results/vectors-0.1.23-20260925/`.

## Gates

1. x86: clean Java 25 build, formatting, recall regression, compliance, dependency locks and workflow
   validation; stage Maven publications and inspect coordinates/license/dependencies.
2. Run existing Q6_K/Q4_K bit-equality and cold/hot JIT tests; explicitly rerun core tests with
   `vectors.maxBits=128` and the forced scalar provider passed into forked test JVMs.
3. ARM: core tests on real AArch64 Java 25, including the Q6_K contracts. A forced 128-bit x86 run
   alone is not evidence of ARM execution.
4. Validate notebooks against the staged 0.1.23 artifacts, plus documentation build and SBOM.
5. Exercise Studio browser workflows (collections, document data, pagination, PCA/t-SNE/UMAP,
   inspector), its integration tests and the UI demos against the candidate libraries. The
   application integration gate is required even though Studio is outside Maven publication.
6. A release is ready only after these gates pass on the exact candidate source. Publication and
   downstream dependency updates must identify the immutable released version.

No performance improvement is claimed by this patch. The correctness correction changes scalar
low bits; the existing >=256-bit SIMD implementation is unchanged. Broader Models native/device
and whole-model parity are downstream gates, not established by Vectors kernel tests.

## Experimental branch disposition

- PR #74, `exp/band-gemm`: retain as a rejected experiment. Recorded token identity missed its
  acceptance gate; do not merge it into this patch release.
- PR #75, `exp/kquant-dequant-simd`: retain as a rejected experiment. Its 256-bit batch-512
  regression exceeded the preregistered ceiling; do not enable it by default.
- Keep all unique experimental logs and worktrees until their evidence is archived and ownership
  is reviewed. No worktree/branch deletion is part of this release.
