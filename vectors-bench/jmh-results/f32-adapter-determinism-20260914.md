# Deterministic F32 adapter execution

Date: 2026-09-14

## Question

Can the Java Vector API execute small F32 LoRA matrices with identical results before and after
HotSpot compilation, without an external inference runtime, a native kernel, or a model-load
prewarm?

The question arose during Qwen 3 Activated-LoRA qualification. An identical first request produced
a different classifier score after the mapped F32 kernel became hot. A standalone fresh-JVM probe
reproduced the change without model state, cache reuse, or parallel execution. At 128 bits the
first differing output was `1.4212487` versus `1.4212488`; at 256 bits it was `1.4212481` versus
`1.4212482`.

## Cause and controls

The mapped kernel ended each vector accumulator with `FloatVector.reduceLanes(ADD)`. Java does not
specify a lane-addition order for this associative reduction, and the interpreted and compiled
Vector API paths used different legal trees. A one-ULP projection difference can amplify across a
transformer and move an Activated-LoRA routing score.

The retained design therefore separates two use cases:

- general mapped F32 model weights keep the established fast reduction;
- small immutable F32 execution weights can be copied once into `F32ExecutionMatrix`, whose
  single-input path uses a fixed shuffle/add tree and four-row unrolling.

Fresh JVM tests cover the real LoRA up/down shapes at 64-, 128-, and 256-bit vector ceilings. They
compare the first invocation with the result after 256 invocations using raw float bits. The new
foreign-memory factory also verifies little-endian decoding and ownership independently of the
source segment.

## Environment and protocol

- Host: Intel Core i7-9750H, macOS Darwin 25.6.0, 6 physical / 12 logical CPUs
- Runtime: Temurin/OpenJDK 25.0.3, HotSpot C2, 256-bit preferred Vector API species
- JMH: 1.37, three forks, three 1-second warmups, five 1-second measurements
- Shapes: rank-up `32x2048` and rank-down `2048x32`
- Comparison: mapped F32 production kernel versus owned deterministic F32 execution matrix
- Raw result SHA-256: `b25b619a92a85e0557e50158eb06278676b14de3ba59d99957da813f88346ad0`

## Result

Average time per matrix-vector product:

| Shape | Mapped F32 | Owned deterministic F32 | Change |
| ---: | ---: | ---: | ---: |
| `32x2048` | 3.567 +/- 0.112 us | 3.944 +/- 0.138 us | 10.57% slower |
| `2048x32` | 12.084 +/- 0.528 us | 7.718 +/- 0.296 us | 36.13% faster |
| Combined adapter pair | 15.651 us | 11.662 us | 25.49% faster |

The fixed tree adds work to the rank-up reduction, but owned heap access and four-row unrolling more
than recover it on the rank-down projection. The complete adapter pair is both deterministic and
faster on this host. This result does not claim the same percentages on Apple Silicon; the
fresh-JVM correctness gate covers its 128-bit width, while model qualification must measure the
complete host envelope.

## Decision and JVM request

Retain the owned deterministic F32 execution layout for immutable low-rank adapter weights. Remove
the Models-level prewarm after Models adopts this API. Do not change the general mapped F32 kernel
and do not introduce a Rust shim.

The JVM opportunity is an explicitly ordered Vector API reduction, or a guarantee that one
reduction expression has the same arithmetic tree before and after compilation. The current API
makes reproducible inference choose between an unspecified fast reduction and a manually expressed
tree. This experiment shows that standard Java can make the right trade for small adapter matrices,
but the missing primitive still complicates portable deterministic numerical code.
