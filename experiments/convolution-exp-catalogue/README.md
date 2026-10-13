# Certified float exponential for convolution

The production helper is `ReproducibleScalarExp.exp(float)`. It uses scalar Math.exp only inside the closed magnitude interval [2^-28,104] and outside conservative exception buckets. Every other input takes StrictMath directly. The convolution retains published full-width float/FMA grouping and history-update order, then applies SiLU in place without another buffer.

## Certificate

Let R be real exp(x), S=StrictMath.exp(x), and V=Math.exp(x). StrictMath.exp has mandated fdlibm5.3 semantics across conforming JVMs (StrictMath class contract); this fixes the reference S and therefore the catalogue across hosts. Both double methods have a one-ulp accuracy contract. For the enumerated inputs, results are positive normal finite doubles. Therefore |V-S| <=2ulp(R) <=4ulp(S), allowing a binade crossing. Eight neighboring positive double bit patterns on either side of S span at least4ulp(S), even across a power-of-two boundary. If both interval endpoints narrow to the same float, every allowed V has the same float cast as S. Retain every input whose endpoints differ, whether or not a particular Math implementation disagrees.

`EnumerateExpExceptions.java` visits all580,911,106 signed input bit patterns with magnitude in the closed interval [2^-28,104]. Its frozen eight-neighbor test produces exactly the15 entries in `exceptions.jsonl`. The first qualifying enumeration receipt is `enumeration.log`; original source/protocol freeze c90d4329 and complete experimental evidence6a3016fc remain in the repository history. Inputs outside this interval, all nonfinite inputs, signed zeros, and subnormals use StrictMath in production, so no analytic underflow-sign assumption is needed.

The fixed boolean[2048] table uses `(raw * 0x9e3779b9) >>> 21`. Every catalogue bucket is marked. False positives deliberately use StrictMath; false negatives are forbidden. Hash/table parameters were fixed before benchmarking and not tuned to model outputs. The helper's sampled unit tests are regression coverage, not a replacement for complete catalogue enumeration or compiled-candidate validation.

## Reproduce

Compile/run the generator with JDK25 (no model weights required), capture its JSONL output and compare the entire catalogue. Run the standalone compiled-candidate helper proof separately, against the exact production JAR, then the actual convolution/history tests at supported SIMD widths and FMA settings. Keep source/JAR hashes with all results. Runtime exactness, cold/warm allocation, latency, and whole-model release gates remain separate checks.

Reference portability contract: https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/StrictMath.html (exp is among the required fdlibm operations).

Official accuracy contracts: https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/Math.html#exp(double) and https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/StrictMath.html#exp(double).
