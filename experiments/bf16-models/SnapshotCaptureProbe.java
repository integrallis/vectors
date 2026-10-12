package com.integrallis.models.backend.purejava;

import java.util.ArrayList;
import java.util.Arrays;

/** Reproduces a backend returning one mutable logits buffer across decode steps. */
final class SnapshotCaptureProbe {
  public static void main(String[] args) {
    float[] shared = {1.0f, 2.0f, 3.0f};
    var snapshots = new ArrayList<float[]>(2);
    long[] cost = new long[2];
    Bf16ModelProof.captureSnapshot(shared, snapshots, cost);
    Arrays.fill(shared, -4.0f);
    Bf16ModelProof.captureSnapshot(shared, snapshots, cost);
    Arrays.fill(shared, 0.0f);
    if (!Arrays.equals(snapshots.get(0), new float[] {1.0f, 2.0f, 3.0f})
        || !Arrays.equals(snapshots.get(1), new float[] {-4.0f, -4.0f, -4.0f}))
      throw new AssertionError("later backend writes changed captured logits");
    if (cost[0] <= 0 || cost[1] <= 0)
      throw new AssertionError("snapshot observation cost was not recorded");
    System.out.println("SNAPSHOT_CAPTURE_OWNERSHIP_PASS");
  }
}
