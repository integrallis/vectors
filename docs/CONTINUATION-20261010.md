# Construction continuation, 2026-10-10

Base: `6858da43688aae6952a6fdee6fb3b2eacf6d7605` (released Vectors 0.1.28 main).

This branch preserves two previously untracked tests from the detached
`vectors-default-baseline-20261003` worktree. It is deliberately **not ready to merge**.
The original files remain untouched in that worktree and in the workspace audit backup.

The tests compile. Three cases ran: input validation passed; the segment-view expectation and
bounded-task expectation failed. Current append submits 400 tasks for four workers in the test;
`SuccessorVectors.asVectors()` does not expose segment views. These tests propose optimization
contracts; their failure alone does not establish a broken public API or a speed improvement.

The recovered task-count assertion originally included a subsequent test-only submission. It now
checks the construction count before that submission. Formatting was applied to the recovered files.

```sh
./gradlew :vectors-db:test --tests '*SuccessorSegmentViewTest' \
  :vectors-hnsw:test --tests '*BoundedConstructionTasksTest' --continue
```

Next: implement and test the proposed bounds/ownership behavior; run the complete affected suites;
measure both arms with identical inputs on the documented benchmark host. Do not merge speculative
SQ8 construction or claim an improvement from these tests.

The older construction experiment remains available locally at
`refs/archive/audit-20261010/heads/backup/original-work-before-integration-20261003`
(`faab491e`; use the full SHA in the workspace preservation manifest). It contains exploratory code,
not a qualified replacement for main. Other rejected Q4 range APIs and historical A/B baselines
were archived rather than reintroduced.
