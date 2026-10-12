package com.integrallis.models.backend.purejava;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;

/** Separate processes select immutable library artifacts, never an inference-route flag. */
public final class Bf16ModelProof {
  static final ObjectMapper JSON = new ObjectMapper();
  static final com.sun.management.ThreadMXBean ALLOCATION =
      (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();

  static {
    if (!ALLOCATION.isThreadAllocatedMemorySupported())
      throw new AssertionError("Java allocation accounting unavailable");
    ALLOCATION.setThreadAllocatedMemoryEnabled(true);
  }

  static void emit(Map<String, ?> row) throws Exception {
    System.out.println(JSON.writeValueAsString(row));
  }

  static String digest(float[] values) throws Exception {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      if (!Float.isFinite(value)) throw new AssertionError("non-finite logit");
      bytes.putInt(Float.floatToRawIntBits(value));
    }
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes.array()));
  }

  static int argmax(float[] values) {
    int best = 0;
    for (int n = 1; n < values.length; n++) if (values[n] > values[best]) best = n;
    return best;
  }

  static Map<String, Object> request(
      BatchInferenceBackend backend, JsonNode task, int repeat, int steps) throws Exception {
    int[] input = backend.tokenizer().encode(task.path("prompt").asText());
    List<String> hashes = new ArrayList<>();
    List<float[]> snapshots = new ArrayList<>();
    List<Integer> tokens = new ArrayList<>();
    long prefill, decode = 0;
    float[] initial;
    long allocatedBefore = ALLOCATION.getTotalThreadAllocatedBytes();
    long requestStart = System.nanoTime();
    try (InferenceSession session = backend.openSession()) {
      long start = System.nanoTime();
      float[] logits = backend.prefill(session, input, 0);
      prefill = System.nanoTime() - start;
      initial = logits;
      snapshots.add(logits);
      for (int n = 0; n < steps; n++) {
        int token = argmax(logits);
        tokens.add(token);
        start = System.nanoTime();
        logits = backend.forward(session, token, input.length + n);
        decode += System.nanoTime() - start;
        snapshots.add(logits);
      }
    }
    long requestNanos = System.nanoTime() - requestStart;
    long javaAllocatedBytes = ALLOCATION.getTotalThreadAllocatedBytes() - allocatedBefore;
    if (allocatedBefore < 0 || javaAllocatedBytes < 0)
      throw new AssertionError("invalid allocation counter");
    for (float[] snapshot : snapshots) hashes.add(digest(snapshot));
    String prediction = "";
    float best = -Float.MAX_VALUE;
    for (int n = 0; n < task.path("labels").size(); n++) {
      int[] label = backend.tokenizer().encode(" " + (char) ('A' + n));
      float score = initial[label[label.length - 1]];
      if (score > best) {
        best = score;
        prediction = task.path("labels").get(n).asText();
      }
    }
    Map<String, Object> row = new LinkedHashMap<>();
    row.put("kind", "request");
    row.put("id", task.path("id").asText());
    row.put("dataset", task.path("dataset").asText());
    row.put("repeat", repeat);
    row.put("promptTokens", input.length);
    row.put("inputTokens", input);
    row.put("prefillNanos", prefill);
    row.put("decodeNanos", decode);
    row.put("javaAllocatedBytes", javaAllocatedBytes);
    row.put("modelNanos", prefill + decode);
    row.put("requestNanos", requestNanos);
    row.put("greedyTokens", tokens);
    row.put("fullLogitSha256", hashes);
    row.put("prediction", prediction);
    row.put("expected", task.path("expected").asText());
    emit(row);
    return row;
  }

  static BatchInferenceBackend load(Path model, Path nativePath, int batch) throws Exception {
    Map<String, String> settings = new HashMap<>();
    settings.put("models.purejava.maxContextLength", "2048");
    if (batch > 0) settings.put("models.purejava.prefillBatchSize", Integer.toString(batch));
    var configuration =
        new BackendConfiguration(
            Map.of("bf16-qualification-context-budget", "2048"), settings, List.of());
    return model.toString().endsWith(".gguf")
        ? RustFfmBackend.load(model, nativePath, configuration)
        : PureJavaBackend.load(model, configuration);
  }

  static float[][] snapshots(BatchInferenceBackend backend, int[] input, int steps)
      throws Exception {
    float[][] results = new float[steps][];
    try (InferenceSession session = backend.openSession()) {
      float[] logits = backend.prefill(session, input, 0);
      for (int step = 0; step < steps; step++) {
        digest(logits);
        results[step] = logits.clone();
        if (step + 1 < steps)
          logits = backend.forward(session, argmax(logits), input.length + step);
      }
    }
    return results;
  }

  static void fixture(Path model, Path nativePath) throws Exception {
    boolean bonsai = model.toString().endsWith(".gguf");
    int[] expected = {33075, 888, 279, 15217, 5388, 13, 271, 760};
    float[] reference = {
      7.1711001f,
      10.372878f,
      6.2867732f,
      6.8912621f,
      4.0959649f,
      4.4131360f,
      7.8144817f,
      7.5291200f,
      7.0460711f,
      8.0585680f,
      5.2136788f,
      6.7079167f,
      8.3307886f,
      11.508839f,
      5.7097855f,
      6.6055269f
    };
    try (var backend = load(model, nativePath, 0)) {
      emit(Map.of("kind", "diagnostics", "value", backend.diagnostics()));
      if (bonsai
          && !"true".equals(backend.diagnostics().environment().get("native-quantized-decode")))
        throw new AssertionError("native default projection route inactive");
      int[] tokens =
          bonsai
              ? backend.tokenizer().encode("The quick brown fox")
              : backend
                  .tokenizer()
                  .encodeControl(
                      "<|im_start|>user\n"
                          + "Name one JVM language.<|im_end|>\n"
                          + "<|im_start|>assistant\n");
      int[] pinnedTokens =
          bonsai
              ? new int[] {760, 3841, 13477, 37550}
              : new int[] {
                151644, 872, 198, 675, 825, 72379, 4128, 13, 151645, 198, 151644, 77091, 198
              };
      if (!Arrays.equals(tokens, pinnedTokens))
        throw new AssertionError("pinned tokenizer mismatch");
      if (backend.diagnostics().optimizations().stream()
          .noneMatch(
              o -> o.id().equals("batched-prefill") && o.status().toString().equals("ENABLED")))
        throw new AssertionError("public batched prefill route inactive");
      List<String> initialHashes = null;
      for (int repeat = 0; repeat < 3; repeat++) {
        float[][] values = snapshots(backend, tokens, bonsai ? 8 : 3);
        if (!bonsai && values[0].length != 151936)
          throw new AssertionError("Qwen vocabulary mismatch");
        if (bonsai) {
          for (int step = 0; step < expected.length; step++)
            if (argmax(values[step]) != expected[step])
              throw new AssertionError(
                  "Bonsai oracle token "
                      + step
                      + " repeat "
                      + repeat
                      + ": "
                      + argmax(values[step]));
        } else {
          for (int n = 0; n < reference.length; n++)
            if (Math.abs(values[0][n] - reference[n]) > 1e-3f)
              throw new AssertionError("Qwen pinned Transformers logit " + n + ": " + values[0][n]);
          if (argmax(values[0]) != 3966 || Math.abs(values[0][3966] - 20.877668f) > 1e-3f)
            throw new AssertionError("Qwen reference winner mismatch");
        }
        List<String> hashes = new ArrayList<>();
        for (float[] value : values) hashes.add(digest(value));
        emit(Map.of("kind", "oracle", "repeat", repeat, "fullLogitSha256", hashes));
        if (initialHashes != null && !initialHashes.equals(hashes))
          throw new AssertionError("fresh session changed after compilation");
        initialHashes = hashes;
      }
      String implicit = digest(backend.prefill(tokens, 0));
      backend.reset();
      String reset = digest(backend.prefill(tokens, 0));
      emit(
          Map.of(
              "kind",
              "implicit-reset",
              "expected",
              initialHashes.getFirst(),
              "implicit",
              implicit,
              "reset",
              reset));
      if (!initialHashes.getFirst().equals(implicit) || !implicit.equals(reset))
        throw new AssertionError("implicit session reset changed logits");
    }
    String prompt =
        "Explain why independent requests must not share recurrent state. A second request should"
            + " start with its own history and produce the same answer when repeated.";
    float[][] single;
    int[] tokens;
    try (var backend = load(model, nativePath, 1)) {
      tokens = backend.tokenizer().encode(prompt);
      if (tokens.length <= 16) throw new AssertionError("missing chunk boundary");
      single = snapshots(backend, tokens, 3);
    }
    try (var backend = load(model, nativePath, 16)) {
      float[][] batch = snapshots(backend, tokens, 3);
      for (int step = 0; step < 3; step++) {
        String before = digest(single[step]), after = digest(batch[step]);
        emit(
            Map.of(
                "kind",
                "batch-parity",
                "step",
                step,
                "expected",
                before,
                "actual",
                after,
                "tokens",
                tokens.length));
        if (!before.equals(after))
          throw new AssertionError("whole-vocabulary batch parity " + step);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    String phase = args[0];
    Path model = Path.of(args[1]), nativePath = Path.of(args[2]);
    emit(
        Map.of(
            "kind",
            "runtime-origin",
            "bf16OrdinaryReduction",
            com.integrallis.vectors.core.PanamaConstants.HAS_SVE ? "inline-sve" : "direct-helper",
            "bf16",
            com.integrallis.vectors.core.BFloat16Matrix.class
                .getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .toString(),
            "models",
            PureJavaBackend.class.getProtectionDomain().getCodeSource().getLocation().toString()));
    if (phase.equals("fixture")) {
      fixture(model, nativePath);
      return;
    }
    JsonNode tasks = JSON.readTree(Path.of(args[3]).toFile());
    int pair = Integer.parseInt(args[4]);
    long start = System.nanoTime();
    try (var backend = load(model, nativePath, 0)) {
      emit(
          Map.of(
              "kind",
              "load",
              "nanos",
              System.nanoTime() - start,
              "diagnostics",
              backend.diagnostics(),
              "phase",
              phase));
      List<JsonNode> jevbench = new ArrayList<>(), snips = new ArrayList<>();
      for (var task : tasks) {
        if (task.path("dataset").asText().equals("jevbench")) jevbench.add(task);
        else if (task.path("dataset").asText().equals("snips")) snips.add(task);
        else throw new AssertionError("undeclared dataset");
      }
      List<JsonNode> timed = new ArrayList<>();
      for (int n = 0; n < 15; n++) {
        timed.add(jevbench.get(n));
        timed.add(snips.get(n));
      }
      request(backend, tasks.get(0), -4, 8);
      for (int n = 0; n < 3; n++) request(backend, timed.get(n), -3 + n, 8);
      for (int n = 0; n < 3; n++) request(backend, timed.get((pair * 3 + n) % timed.size()), n, 8);
    }
  }
}
