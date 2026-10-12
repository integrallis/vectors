package experiments.activation;

import com.integrallis.vectors.core.VectorUtil;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
public class ActivationPublicBenchmark {
  @Param({"32", "4096", "16384"}) public int size;
  float[] input, up, output, history, weights;
  @Setup public void setup() {
    var random = new SplittableRandom(738126);
    input = new float[size]; up = new float[size]; output = new float[size];
    history = new float[2*size]; weights = new float[3*size];
    for (int i = 0; i < size; i++) {
      input[i] = (float)random.nextDouble(-8,8);
      up[i] = (float)random.nextDouble(-2,2);
    }
    for (int i = 0; i < weights.length; i++) weights[i] = (float)random.nextDouble(-.25,.25);
  }
  @Benchmark public float[] swiGlu() {
    VectorUtil.swiGlu(output,0,input,0,up,0,size);
    return output;
  }
  @Benchmark public float[] causalSilu() {
    System.arraycopy(input,0,output,0,size);
    VectorUtil.causalDepthwiseConv1dSilu(output,0,history,0,weights,0,size,3);
    return output;
  }
}
