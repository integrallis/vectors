# Bonsai activation contract coverage

These are local kernel correctness checks, not model qualification or performance measurements.
The new probe covers the actual 48-head sigmoid/softplus shape and twelve consecutive four-tap
convolution updates, including persistent history, offsets, scalar tails and untouched guards.
It checks the first sequence against an independent scalar replay of the existing arithmetic
contract, then repeats the sequence 2,000 times across JIT compilation.

The softplus oracle deliberately preserves vector float intermediates, which differ from the
scalar-tail log1p contract for negative inputs. It does not silently change that production
contract. Convolution replay preserves fused vector lanes and unfused scalar tails.

Ten fresh JVMs cover default and 64/128/256/512 width ceilings, with default FMA settings and
both FMA toggles disabled. The recorded host prefers 256 bits: the 512 ceiling exercises 256,
not native 512-bit execution.

All ten configurations pass both the published 0.1.29 JAR and broad candidate 6893a587 on local
Temurin 25.0.3 x86_64. This is **not** a red/green reproduction on this host. The existing Linux
real-model failures remain separate red evidence; Linux/ARM execution of these new tests is
still required. No compiler flags were added to manufacture a local failure.

`released-receipt.json` retains exact commands and the published JAR hash. `manifest.json`
records candidate source, compiled class hashes and evidence hashes. JUnit and per-fork logs
retain the effective species, FMA settings, JDK and architecture.
