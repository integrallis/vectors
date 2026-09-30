/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufBatchedMatmulKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Layer-by-layer hidden-state probe for pre-registration 4. Launched on the classpath of the
 * model-check harness (ModelCheck classes) plus the Models {@code models-bench} install.
 *
 * <pre>
 * LayerProbe --model M --window W --recorded cont-base-arm-integer.json
 *            --arm-continuations cont-base-arm-ARM.json --ids id1,id2,id3
 *            --condition pipeline-cache|fresh --out-dir D [--all-positions true]
 * </pre>
 *
 * The kernel arm is whatever {@code -Dvectors.gguf.batchedMatmulKernel} selects for this JVM; the
 * output records it. Per target prompt it writes {@code D/CONDITION-ARM-ID.json} with the last
 * prompt position's vectors at stages {@code embedding} (recomputed: token row times
 * embeddingScale; the observer does not expose it), {@code layer.0..layer.N-1} (the forward pass's
 * layerObserver), {@code final_norm} (the pass's xNorm after prefill: the vector the LM head read)
 * and {@code logits} (the prefill return value). With {@code --all-positions true} (fresh only) it
 * also writes {@code D/fresh-ARM-ID.f32}: little-endian float32 of shape [N+1, T, dim], index 0
 * the recomputed embeddings, index L+1 the output of layer L, for every prompt position.
 *
 * <p>pipeline-cache replays ModelCheck's prefill sequence: the recorded cases in order, each
 * rewound to its longest shared token prefix with the previous prompt (GenerationLoop's
 * reusablePrefixLength, capped at length - 1) and only the suffix prefilled. fresh resets and
 * prefills the whole prompt, once without the observer and once with it, and records whether the
 * two logits vectors are bit-identical.
 */
public final class LayerProbe {
  static final ObjectMapper JSON = new ObjectMapper();

  public static void main(String[] args) throws Exception {
    Map<String, String> options = new HashMap<>();
    for (int i = 0; i < args.length; i += 2) {
      if (!args[i].startsWith("--") || i + 1 >= args.length) {
        throw new IllegalArgumentException("expected --key value at " + args[i]);
      }
      options.put(args[i].substring(2), args[i + 1]);
    }
    String condition = ModelCheck.required(options, "condition");
    if (!condition.equals("pipeline-cache") && !condition.equals("fresh")) {
      throw new IllegalArgumentException("--condition must be pipeline-cache or fresh");
    }
    boolean allPositions = Boolean.parseBoolean(options.getOrDefault("all-positions", "false"));
    if (allPositions && !condition.equals("fresh")) {
      throw new IllegalArgumentException("--all-positions needs --condition fresh");
    }
    String arm = System.getProperty("vectors.gguf.batchedMatmulKernel", "default");
    Path outDir = Path.of(ModelCheck.required(options, "out-dir"));
    Files.createDirectories(outDir);
    Set<String> targets =
        new LinkedHashSet<>(Arrays.asList(ModelCheck.required(options, "ids").split(",")));

    JsonNode window = JSON.readTree(Path.of(ModelCheck.required(options, "window")).toFile());
    JsonNode recorded = JSON.readTree(Path.of(ModelCheck.required(options, "recorded")).toFile());
    JsonNode armContinuations =
        JSON.readTree(Path.of(ModelCheck.required(options, "arm-continuations")).toFile());
    JsonNode suite = null;
    for (JsonNode candidate : window.path("suites")) {
      if (candidate.path("name").asText().equals(recorded.path("suite").asText())) {
        suite = candidate;
      }
    }
    if (suite == null) {
      throw new IllegalStateException("window has no suite " + recorded.path("suite").asText());
    }
    Map<String, JsonNode> items = new HashMap<>();
    for (JsonNode item : suite.path("cases")) {
      items.put(item.path("id").asText(), item);
    }
    Map<String, JsonNode> armCases = new HashMap<>();
    for (JsonNode c : armContinuations.path("cases")) {
      armCases.put(c.path("id").asText(), c);
    }

    // Walk order: the recorded cases in the order ModelCheck generated them.
    List<JsonNode> order = new ArrayList<>();
    for (JsonNode rec : recorded.path("cases")) {
      order.add(rec);
    }
    int lastTarget = -1;
    for (int i = 0; i < order.size(); i++) {
      if (targets.contains(order.get(i).path("id").asText())) {
        lastTarget = i;
      }
    }
    for (String id : targets) {
      if (order.stream().noneMatch(rec -> rec.path("id").asText().equals(id))) {
        throw new IllegalArgumentException("target " + id + " is not in the recorded cases");
      }
    }

    try (PureJavaBackend backend = PureJavaBackend.load(Path.of(ModelCheck.required(options, "model")))) {
      Tokenizer tokenizer = backend.tokenizer();
      LlamaForwardPass pass = forwardPass(backend);
      LlamaConfig config = (LlamaConfig) read(pass, "config");
      LlamaWeights weights = (LlamaWeights) read(pass, "weights");
      int layers = config.numLayers();
      int dim = config.embeddingDim();
      int batchCapacity = (Integer) read(pass, "prefillBatchCapacity");
      System.out.printf(
          "LAYERPROBE arm=%s condition=%s layers=%d dim=%d prefillBatchCapacity=%d kernel=%s%n",
          arm, condition, layers, dim, batchCapacity,
          VectorUtil.runtimeCapabilities().ggufBatchedMatmulKernel());

      int[] previous = null;
      for (int index = 0; index <= lastTarget; index++) {
        JsonNode rec = order.get(index);
        String id = rec.path("id").asText();
        JsonNode item = items.get(id);
        if (item == null) {
          throw new IllegalStateException("window has no case " + id);
        }
        ModelPrompt prompt = ModelCheck.prompt(item, ModelCheck.BASE_INSTRUCTION);
        StringBuilder text = new StringBuilder();
        for (ModelPrompt.Segment segment : prompt.segments()) {
          text.append(segment.text());
        }
        String sha = ModelCheck.sha256(text.toString());
        if (!sha.equals(rec.path("promptTextSha256").asText())) {
          throw new IllegalStateException("prompt SHA-256 mismatch for " + id + ": " + sha);
        }
        int[] tokens = tokenizer.encode(prompt);
        if (tokens.length != rec.path("promptTokens").asInt()) {
          throw new IllegalStateException(
              "prompt token count mismatch for " + id + ": " + tokens.length + " vs recorded "
                  + rec.path("promptTokens").asInt());
        }
        boolean target = targets.contains(id);
        if (condition.equals("fresh") && !target) {
          continue;
        }

        Capture capture = target ? new Capture(layers, dim, tokens.length, allPositions) : null;
        Map<String, Object> result = new LinkedHashMap<>();
        float[] logits;
        int start;
        long started;
        long millis;
        if (condition.equals("fresh")) {
          start = 0;
          installObserver(pass, null);
          backend.reset();
          float[] control = backend.prefill(tokens, 0).clone();
          installObserver(pass, capture);
          backend.reset();
          started = System.nanoTime();
          logits = backend.prefill(tokens, 0).clone();
          millis = (System.nanoTime() - started) / 1_000_000L;
          installObserver(pass, null);
          result.put("observerLogitsBitIdentical", Arrays.equals(control, logits));
          result.put("observerLogitsMaxAbsDiff", maxAbsDiff(control, logits));
        } else {
          // GenerationLoop.preparePromptTokens: reuse the shared prefix, else reset.
          int shared = 0;
          if (previous != null) {
            int limit = Math.min(previous.length, tokens.length);
            while (shared < limit && previous[shared] == tokens[shared]) {
              shared++;
            }
            shared = Math.min(shared, tokens.length - 1);
          }
          if (shared == 0) {
            backend.reset();
          } else {
            backend.rewind(shared);
          }
          start = shared;
          installObserver(pass, capture);
          started = System.nanoTime();
          logits = backend.prefill(Arrays.copyOfRange(tokens, shared, tokens.length), shared).clone();
          millis = (System.nanoTime() - started) / 1_000_000L;
          installObserver(pass, null);
          previous = tokens;
          System.out.printf("LAYERPROBE replay %s shared=%d suffix=%d%n", id, shared, tokens.length - shared);
        }
        if (!target) {
          continue;
        }

        float[] xNorm = ((float[]) read(pass, "xNorm")).clone();
        float[] embedding = new float[dim];
        embed(weights, config, tokens[tokens.length - 1], embedding);
        float[] lastLayer = capture.last[layers - 1];
        List<String> missing = new ArrayList<>();
        for (int layer = 0; layer < layers; layer++) {
          if (capture.last[layer] == null) {
            missing.add("layer." + layer);
          }
        }
        if (!missing.isEmpty()) {
          throw new IllegalStateException("observer did not report " + missing + " for " + id);
        }
        float[] recomputedNorm = new float[dim];
        TensorOps.rmsNorm(recomputedNorm, lastLayer, weights.outputNormWeight(), dim, config.rmsNormEps());

        List<String> stages = new ArrayList<>();
        List<float[]> vectors = new ArrayList<>();
        stages.add("embedding");
        vectors.add(embedding);
        for (int layer = 0; layer < layers; layer++) {
          stages.add("layer." + layer);
          vectors.add(capture.last[layer]);
        }
        stages.add("final_norm");
        vectors.add(xNorm);
        stages.add("logits");
        vectors.add(logits);

        int[] top = topK(logits, 5);
        List<Map<String, Object>> topList = new ArrayList<>();
        for (int t : top) {
          Map<String, Object> entry = new LinkedHashMap<>();
          entry.put("token", t);
          entry.put("text", tokenizer.decode(t));
          entry.put("logit", logits[t]);
          topList.add(entry);
        }
        JsonNode armCase = armCases.get(id);
        String recordedFirst =
            armCase == null || armCase.path("fragments").isEmpty()
                ? null
                : armCase.path("fragments").get(0).asText();

        result.put("id", id);
        result.put("arm", arm);
        result.put("condition", condition);
        result.put("walkIndex", index);
        result.put("vectorsKernel", VectorUtil.runtimeCapabilities().ggufBatchedMatmulKernel());
        result.put("promptTextSha256", sha);
        result.put("promptText", text.toString());
        result.put("promptTokens", tokens.length);
        result.put("tokenIds", tokens);
        result.put("prefillStartPosition", start);
        result.put("prefillTokens", tokens.length - start);
        result.put("prefillBatchCapacity", batchCapacity);
        // LlamaForwardPass.prefillBatched: chunks of min(capacity, remaining); a final chunk of one
        // token runs the single-token path. The last position's chunk size decides its band routing.
        int prefilled = tokens.length - start;
        result.put(
            "lastPositionChunkSize",
            batchCapacity <= 1 || prefilled == 1 ? 1 : ((prefilled - 1) % batchCapacity) + 1);
        result.put("prefillMillis", millis);
        result.put("layers", layers);
        result.put("dim", dim);
        result.put("vocab", logits.length);
        result.put(
            "stageProvenance",
            Map.of(
                "embedding", "recomputed: LlamaWeights.embedToken x LlamaConfig.embeddingScale",
                "layer.L", "observed: LlamaForwardPass.layerObserver after decoder layer L",
                "final_norm", "observed: LlamaForwardPass.xNorm after prefill (LM-head input)",
                "logits", "observed: prefill return value (after logit scaling)"));
        result.put("finalNormRecomputedMaxAbsDiff", maxAbsDiff(recomputedNorm, xNorm));
        result.put("nonFiniteValues", nonFinite(vectors));
        result.put("top5", topList);
        result.put("argmaxText", tokenizer.decode(top[0]));
        result.put("recordedFirstFragment", recordedFirst);
        result.put(
            "reproducesRecordedFirstToken",
            recordedFirst != null && recordedFirst.equals(tokenizer.decode(top[0])));
        result.put("stages", stages);
        result.put("vectors", vectors);
        result.put("observedPositions", capture.positionsSeen);
        if (allPositions) {
          Path raw = outDir.resolve(condition + "-" + arm + "-" + id + ".f32");
          for (int p = 0; p < tokens.length; p++) {
            embed(weights, config, tokens[p], capture.all[0][p]);
          }
          writeFloat32(raw, capture.all);
          result.put("allPositionsFile", raw.getFileName().toString());
          result.put("allPositionsShape", new int[] {layers + 1, tokens.length, dim});
          result.put("allPositionsLayout", "index 0 = recomputed embedding, index L+1 = output of layer L");
        }
        result.put("vectorsRouting", GgufBatchedMatmulKernel.routingReport());
        Path out = outDir.resolve(condition + "-" + arm + "-" + id + ".json");
        JSON.writeValue(out.toFile(), result);
        System.out.printf(
            "LAYERPROBE %s start=%d prefill=%d lastChunk=%s millis=%d argmax=%s recordedFirst=%s reproduces=%s%s -> %s%n",
            id, start, tokens.length - start, result.get("lastPositionChunkSize"), millis,
            JSON.writeValueAsString(tokenizer.decode(top[0])),
            JSON.writeValueAsString(recordedFirst),
            result.get("reproducesRecordedFirstToken"),
            result.containsKey("observerLogitsBitIdentical")
                ? " observerBitIdentical=" + result.get("observerLogitsBitIdentical")
                : "",
            out);
      }
    }
    System.out.println("LAYERPROBE routing: " + GgufBatchedMatmulKernel.routingReport());
  }

  /** Receives per-layer states; keeps the last prompt position and optionally every position. */
  static final class Capture {
    final int lastPosition;
    final float[][] last;
    final float[][][] all;
    final int dim;
    int positionsSeen;
    int minPosition = Integer.MAX_VALUE;

    Capture(int layers, int dim, int promptTokens, boolean allPositions) {
      this.lastPosition = promptTokens - 1;
      this.last = new float[layers][];
      this.dim = dim;
      this.all = allPositions ? new float[layers + 1][promptTokens][dim] : null;
    }

    void onLayerComplete(int layer, int position, float[] state, int offset, int length) {
      if (length != dim) {
        throw new IllegalStateException("observer length " + length + " != dim " + dim);
      }
      if (layer == 0) {
        positionsSeen++;
      }
      if (position == lastPosition) {
        last[layer] = Arrays.copyOfRange(state, offset, offset + length);
      }
      if (all != null) {
        System.arraycopy(state, offset, all[layer + 1][position], 0, length);
      }
    }
  }

  static void embed(LlamaWeights weights, LlamaConfig config, int token, float[] out) {
    weights.embedToken(token, out);
    float scale = config.embeddingScale();
    for (int i = 0; i < out.length; i++) {
      out[i] *= scale;
    }
  }

  static LlamaForwardPass forwardPass(PureJavaBackend backend) throws Exception {
    Object decoder = read(backend, "decoder");
    for (Field field : decoder.getClass().getDeclaredFields()) {
      field.setAccessible(true);
      if (field.get(decoder) instanceof LlamaForwardPass pass) {
        return pass;
      }
    }
    throw new IllegalStateException("no LlamaForwardPass inside " + decoder.getClass());
  }

  static Object read(Object owner, String name) throws Exception {
    Field field = owner.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return field.get(owner);
  }

  /** Sets LlamaForwardPass's private final layerObserver (a package-private interface) by proxy. */
  static void installObserver(LlamaForwardPass pass, Capture capture) throws Exception {
    Field field = LlamaForwardPass.class.getDeclaredField("layerObserver");
    field.setAccessible(true);
    Object proxy = null;
    if (capture != null) {
      Class<?> type = field.getType();
      proxy =
          Proxy.newProxyInstance(
              type.getClassLoader(),
              new Class<?>[] {type},
              (self, method, args) -> {
                switch (method.getName()) {
                  case "onLayerComplete" -> {
                    capture.onLayerComplete(
                        (Integer) args[0], (Integer) args[1], (float[]) args[2],
                        (Integer) args[3], (Integer) args[4]);
                    return null;
                  }
                  case "toString" -> {
                    return "layer-probe";
                  }
                  case "hashCode" -> {
                    return System.identityHashCode(self);
                  }
                  case "equals" -> {
                    return self == args[0];
                  }
                  default -> throw new UnsupportedOperationException(method.getName());
                }
              });
    }
    field.set(pass, proxy);
    if (field.get(pass) != proxy) {
      throw new IllegalStateException("layerObserver write did not take effect");
    }
  }

  static void writeFloat32(Path target, float[][][] data) throws Exception {
    try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(target), 1 << 20)) {
      ByteBuffer buffer = ByteBuffer.allocate(data[0][0].length * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (float[][] layer : data) {
        for (float[] row : layer) {
          buffer.clear();
          buffer.asFloatBuffer().put(row);
          out.write(buffer.array(), 0, row.length * 4);
        }
      }
    }
  }

  static double maxAbsDiff(float[] a, float[] b) {
    double max = 0;
    for (int i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs((double) a[i] - b[i]));
    }
    return max;
  }

  static int nonFinite(List<float[]> vectors) {
    int count = 0;
    for (float[] v : vectors) {
      for (float f : v) {
        if (!Float.isFinite(f)) {
          count++;
        }
      }
    }
    return count;
  }

  static int[] topK(float[] values, int k) {
    int[] best = new int[k];
    Arrays.fill(best, -1);
    for (int i = 0; i < values.length; i++) {
      for (int slot = 0; slot < k; slot++) {
        if (best[slot] < 0 || values[i] > values[best[slot]]) {
          System.arraycopy(best, slot, best, slot + 1, k - slot - 1);
          best[slot] = i;
          break;
        }
      }
    }
    return best;
  }
}
