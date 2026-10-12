# BF16 four-row array boundary experiment

This source is isolated and **not approved for release**. Expanding packed BF16 loads and per-row FMA inside the array-only four-row method attempts to stop vector objects escaping in actual Bonsai inference. The reduction tree and hardware FMA choice remain unchanged.

The existing actual-model JFR shows escaped BF16 vectors even though the standalone kernel improved. The new small-kernel allocation is still about 2,305 bytes/op on this local Intel host; this experiment must therefore be assessed in its actual model caller context. Local forced-64 cold/hot shapes 48×5120, 5×37, 8193×37 pass. The unchanged 180-second 512×2048 child timed out under contention and is retained as a failed execution. Small 5×37 forced-128/256 probes pass. Local first Bonsai request matches all nine full-logit hashes of the frozen supported-host runtime; its elapsed time is not a benchmark due contention.

A separate immutable supported-host diagnostic runs two counterbalanced fresh-process pairs, each with one cold, one warm and two measured actual Bonsai requests. It preserves owned full-logit snapshots, ordinary default runtime settings and an uninstrumented control. Limits: one x86 16-vCPU host, 1,200 seconds, $0.50, absolute 22:30 UTC. It cannot substitute for the original ten-pair qualification matrix.

Experimental core SHA256: `4320bd5cb517a891517de41704690d52aa47b77ba0497c49a1cd7c07fd67c331`. Control core: overflow-fixed `79f536b453c5ad6d7339b30e41d8c8c440da3ff613caf5d1bea6f83e40ade1c7`.
