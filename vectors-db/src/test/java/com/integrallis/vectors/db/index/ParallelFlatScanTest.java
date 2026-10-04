package com.integrallis.vectors.db.index;

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ParallelFlatScanTest {
  @Test
  void parallelScoresPreserveOrdinalOrderMetricsAndMutations() {
    var random = new SplittableRandom(635);
    float[][] rows = new float[4099][129];
    float[] query = new float[129];
    for(var row:rows) for(int d=0;d<row.length;d++) row[d]=(float)random.nextDouble(-.08,.08);
    for(int pass=0;pass<3;pass++) {
      for(int d=0;d<query.length;d++) query[d]=(float)random.nextDouble(-.08,.08);
      for(var metric:SimilarityFunction.values()) {
        float[] scores=FlatScanAdapter.scoreAll(query,rows,metric);
        for(int i=0;i<rows.length;i++) assertEquals(metric.compare(query,rows[i]),scores[i],1e-6f);
      }
      Arrays.fill(rows[137],.031f);
    }
  }

  @Test
  void largeSearchPreservesTieOrderAndConcurrentQueryIsolation() throws Exception {
    int n=40001,dim=128;
    float[][] rows=new float[n][];
    float[] best=new float[dim],worst=new float[dim];
    Arrays.fill(best,.1f);Arrays.fill(worst,-.1f);
    for(int i=0;i<n;i++) rows[i]=(i%3==0)?best:worst;
    var large=new FlatScanAdapter();large.build(rows,SimilarityFunction.EUCLIDEAN);
    var small=new FlatScanAdapter();small.build(Arrays.copyOf(rows,40),SimilarityFunction.EUCLIDEAN);
    // All admitted top-k entries occur in the first 40 rows. Subsequent equal scores must never
    // change the bounded heap's tie order; this compares the automatic route to a serial scan.
    try(var executor=Executors.newVirtualThreadPerTaskExecutor()) {
      var jobs=new java.util.ArrayList<java.util.concurrent.Future<?>>();
      for(int task=0;task<16;task++) {
        final float[] q=task%2==0?best:worst;
        jobs.add(executor.submit(()-> {
          var expected=small.search(q,10,100,1f);
          var actual=large.search(q,10,100,1f);
          assertArrayEquals(expected.ordinals(),actual.ordinals());
          assertArrayEquals(expected.scores(),actual.scores());
        }));
      }
      for(var job:jobs) job.get();
    }
    Arrays.fill(best,.2f);
    assertArrayEquals(small.search(worst,10,100,1f).ordinals(),large.search(worst,10,100,1f).ordinals());
  }
}
