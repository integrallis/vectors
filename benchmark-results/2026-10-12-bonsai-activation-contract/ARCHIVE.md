# Archived diagnostic execution files

The completed one-off BF16, convolution, Bonsai and exponential compatibility
workflows and harnesses are outside the release checkout. This removes experimental
execution machinery, not regression tests, reusable benchmarks or scientific evidence.
It does not establish that any inference regression is fixed.

- Original source: `6a8cce71972356d423297223efa8257da9d80a28`.
- Archived files: 48; each original path, size and SHA-256 is recorded in
  `ARCHIVE-MANIFEST.json` inside the archive.
- Archive SHA-256: `d50227a7b7da5225a0c93da592647ea2924ff7222752ae99945abb29f18e4d21`.
- Durable archive: `s3://models-qual-077051030817/evidence/vectors-diagnostics-d50227a7b7da5225a0c93da592647ea2924ff7222752ae99945abb29f18e4d21.tar.gz`.
- Workspace receipt: `audit-artifacts/vectors-diagnostic-archive-20261013/receipt.json`.

Remote archive bytes and every individual file were read back and verified before
removal. Ongoing experiments use separate proof branches and external payloads;
rejected implementations must not return as dormant release code.
