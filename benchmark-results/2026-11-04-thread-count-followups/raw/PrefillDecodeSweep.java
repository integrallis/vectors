// Prefill and decode timing for one model at one thread count.
//
// Deliberately thin: every number printed comes from the library's own GenerationMetrics
// (prefill, timeToFirstToken, decode, decodeTokensPerSecond), not from a stopwatch added here.
// That is what makes this a driver rather than a new benchmark -- the instrumentation under test
// is the same one PrefillProfileCli reports, which is what the cross-validation arm checks.
//
// Run: java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
//        -Dvectors.gguf.threads=N -Dmodels.native.kernels.threads=N \
//        -cp <classpath> PrefillDecodeSweep <prompt-file> <maxTokens> <repeats>
import com.integrallis.models.api.SamplingOptions;
import java.nio.file.Files;
import java.nio.file.Path;
import org.modeljars.ModelJars;

public class PrefillDecodeSweep {

  private static final String MODEL =
      "org.modeljars.huggingface:qwen.qwen2.5-1.5b-instruct-gguf.q4_k_m:2.5.0-q4_k_m.1";

  public static void main(String[] args) throws Exception {
    String prompt = Files.readString(Path.of(args[0]));
    int maxTokens = Integer.parseInt(args[1]);
    int repeats = Integer.parseInt(args[2]);

    System.out.printf(
        "machine: %s %s, %d available processors, JVM %s%n",
        System.getProperty("os.name"),
        System.getProperty("os.arch"),
        Runtime.getRuntime().availableProcessors(),
        System.getProperty("java.version"));
    System.out.printf("threads toggle: %s%n", System.getProperty("vectors.gguf.threads", "default"));

    try (var runtime = ModelJars.openRuntime(MODEL)) {
      var options = SamplingOptions.builder().temperature(0).maxTokens(maxTokens).build();

      // One warm-up generation that is measured and discarded: the first call carries JIT and is
      // never the number anyone wants, and dropping it silently would hide how big it was.
      System.out.printf("warmup (discarded): %s%n", generate(runtime, unique(prompt, 0), options));

      // Each repeat gets a unique leading line. Without it the pipeline's prompt cache serves the
      // shared prefix and prefill collapses to nothing: the first draft of this script reported
      // 3206 tokens prefilled in 72.3 ms, which is 44,337 tok/s and is not a speed any of this
      // hardware can reach. A measurement that beats the machine is a broken measurement, and the
      // cache was measuring itself. Decode is unaffected and needs no such treatment.
      for (int repeat = 1; repeat <= repeats; repeat++) {
        System.out.printf("rep %d: %s%n", repeat, generate(runtime, unique(prompt, repeat), options));
      }
    }
  }

  /** The same prompt behind a run-specific first line, so the prefix cache cannot serve it. */
  private static String unique(String prompt, int run) {
    return "Run identifier " + (9000 + run) + " alpha.\n" + prompt;
  }

  private static String generate(AutoCloseable runtime, String prompt, SamplingOptions options)
      throws Exception {
    var pipeline = ((org.modeljars.ModelJarRuntime) runtime).pipeline();
    String answer = pipeline.generate(prompt, options);
    var m = pipeline.lastGenerationMetrics();
    if (!m.available()) {
      return "metrics unavailable -- nothing reportable";
    }
    double prefillTokensPerSecond =
        m.prefill().toNanos() == 0
            ? 0
            : m.usage().promptTokens() / (m.prefill().toNanos() / 1_000_000_000.0);
    return String.format(
        "promptTokens=%d completionTokens=%d prefill=%.1f ms (%.1f tok/s) "
            + "ttft=%s decode=%.1f ms (%.2f tok/s) total=%.1f ms chars=%d",
        m.usage().promptTokens(),
        m.usage().completionTokens(),
        m.prefill().toNanos() / 1e6,
        prefillTokensPerSecond,
        m.timeToFirstToken().map(d -> String.format("%.1f ms", d.toNanos() / 1e6)).orElse("n/a"),
        m.decode().toNanos() / 1e6,
        m.decodeTokensPerSecond(),
        m.total().toNanos() / 1e6,
        answer.length())
        + " promptCache=" + m.promptCache();
  }
}
