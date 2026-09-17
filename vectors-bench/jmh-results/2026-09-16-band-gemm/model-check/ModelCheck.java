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
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.chat.GraniteDocumentsPrompt;
import com.integrallis.vectors.core.GgufBatchedMatmulKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Model-level check for the band/integer batch dispatch (pre-registration 2), launched as a
 * single-file source program on the classpath of a Models {@code models-bench} install built with
 * {@code --include-build} against this Vectors branch.
 *
 * <pre>
 * calibrate      --model M --text T --target-tokens N --out P
 *     Longest character prefix of T that the model's tokenizer encodes to at most N tokens.
 * continuations  --model M --window W --suite S --limit 20 --max-tokens 64 --variant base-arm|open
 *                --out F.json
 *     Greedy continuations of the first cases of a frozen answerability window. base-arm renders
 *     exactly ActivatedAnswerabilityQualificationCli's base arm (documents system message with the
 *     one-word instruction, conversation turns, assistant marker) through the public
 *     GraniteDocumentsPrompt renderer; open omits the instruction. No adapter is loaded.
 * </pre>
 *
 * Every output records the active Vectors kernel mode, so an arm is self-describing.
 */
public final class ModelCheck {
  static final String BASE_INSTRUCTION = "Answer with exactly one word: answerable or unanswerable";
  static final ObjectMapper JSON = new ObjectMapper();

  public static void main(String[] args) throws Exception {
    if (args.length == 0) {
      throw new IllegalArgumentException("usage: calibrate|continuations --key value ...");
    }
    Map<String, String> options = new HashMap<>();
    for (int i = 1; i < args.length; i += 2) {
      if (!args[i].startsWith("--") || i + 1 >= args.length) {
        throw new IllegalArgumentException("expected --key value at " + args[i]);
      }
      options.put(args[i].substring(2), args[i + 1]);
    }
    switch (args[0]) {
      case "calibrate" -> calibrate(options);
      case "continuations" -> continuations(options);
      default -> throw new IllegalArgumentException("unknown command " + args[0]);
    }
  }

  static void calibrate(Map<String, String> options) throws Exception {
    String text = Files.readString(Path.of(required(options, "text")));
    int target = Integer.parseInt(required(options, "target-tokens"));
    try (PureJavaBackend backend = PureJavaBackend.load(Path.of(required(options, "model")))) {
      Tokenizer tokenizer = backend.tokenizer();
      if (tokenizer.encode(text).length < target) {
        throw new IllegalStateException("source text encodes to fewer than " + target + " tokens");
      }
      int low = 1;
      int high = text.length();
      while (low < high) {
        int middle = (low + high + 1) >>> 1;
        if (tokenizer.encode(text.substring(0, middle)).length <= target) {
          low = middle;
        } else {
          high = middle - 1;
        }
      }
      String prompt = text.substring(0, low);
      Files.writeString(Path.of(required(options, "out")), prompt);
      System.out.printf(
          "calibrated prompt chars=%d tokens=%d target=%d%n",
          prompt.length(), tokenizer.encode(prompt).length, target);
    }
  }

  static void continuations(Map<String, String> options) throws Exception {
    Path windowPath = Path.of(required(options, "window"));
    JsonNode window = JSON.readTree(windowPath.toFile());
    String suiteName = required(options, "suite");
    int limit = Integer.parseInt(required(options, "limit"));
    int maxTokens = Integer.parseInt(required(options, "max-tokens"));
    String variant = required(options, "variant");
    if (!variant.equals("base-arm") && !variant.equals("open")) {
      throw new IllegalArgumentException("--variant must be base-arm or open");
    }
    JsonNode suite = null;
    for (JsonNode candidate : window.path("suites")) {
      if (suiteName.equals(candidate.path("name").asText())) {
        suite = candidate;
      }
    }
    if (suite == null) {
      throw new IllegalArgumentException("window has no suite " + suiteName);
    }
    SamplingOptions sampling =
        SamplingOptions.builder().temperature(0).maxTokens(maxTokens).build();
    List<Map<String, Object>> results = new ArrayList<>();
    try (PureJavaBackend backend = PureJavaBackend.load(Path.of(required(options, "model")));
        InferencePipeline pipeline = new InferencePipeline(backend)) {
      int index = 0;
      for (JsonNode item : suite.path("cases")) {
        if (index++ >= limit) {
          break;
        }
        ModelPrompt prompt = prompt(item, variant.equals("base-arm") ? BASE_INSTRUCTION : null);
        StringBuilder text = new StringBuilder();
        for (ModelPrompt.Segment segment : prompt.segments()) {
          text.append(segment.text());
        }
        List<String> fragments = new ArrayList<>();
        Object[] completion = new Object[2];
        long started = System.nanoTime();
        pipeline.generate(
            prompt,
            sampling,
            new TokenStream() {
              @Override
              public void onToken(String token) {
                fragments.add(token);
              }

              @Override
              public void onComplete() {}

              @Override
              public void onComplete(GenerationUsage usage, StopReason stopReason) {
                completion[0] = usage;
                completion[1] = stopReason;
              }

              @Override
              public void onError(Throwable failure) {
                throw new IllegalStateException(failure);
              }
            });
        long millis = (System.nanoTime() - started) / 1_000_000L;
        GenerationUsage usage = (GenerationUsage) completion[0];
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", item.path("id").asText());
        result.put("label", item.path("label").asText());
        result.put("promptTextSha256", sha256(text.toString()));
        result.put("promptTokens", usage == null ? -1 : usage.promptTokens());
        result.put("completionTokens", usage == null ? fragments.size() : usage.completionTokens());
        result.put("stopReason", String.valueOf(completion[1]));
        result.put("output", String.join("", fragments));
        result.put("fragments", fragments);
        result.put("millis", millis);
        results.add(result);
        System.out.printf(
            "%s tokens=%s stop=%s millis=%d output=%s%n",
            result.get("id"),
            result.get("completionTokens"),
            result.get("stopReason"),
            millis,
            JSON.writeValueAsString(result.get("output")));
      }
    }
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("variant", variant);
    report.put("suite", suiteName);
    report.put("limit", limit);
    report.put("maxTokens", maxTokens);
    report.put("windowSha256", window.path("windowSha256").asText());
    report.put("windowFileSha256", sha256(Files.readAllBytes(windowPath)));
    report.put("vectorsKernel", VectorUtil.runtimeCapabilities().ggufBatchedMatmulKernel());
    report.put("vectorsRouting", GgufBatchedMatmulKernel.routingReport());
    report.put("capabilities", VectorUtil.runtimeCapabilities().toString());
    report.put("cases", results);
    JSON.writerWithDefaultPrettyPrinter().writeValue(Path.of(required(options, "out")).toFile(), report);
    System.out.println("routing: " + GgufBatchedMatmulKernel.routingReport());
  }

  static ModelPrompt prompt(JsonNode item, String instruction) {
    List<String> documents = new ArrayList<>();
    for (JsonNode document : item.path("documents")) {
      documents.add(document.path("text").asText());
    }
    ModelPrompt.Builder builder =
        GraniteDocumentsPrompt.appendSystem(ModelPrompt.builder(), documents, instruction);
    for (JsonNode message : item.path("messages")) {
      GraniteDocumentsPrompt.appendTurn(
          builder, message.path("role").asText(), message.path("text").asText());
    }
    return GraniteDocumentsPrompt.finish(builder);
  }

  static String sha256(String text) throws Exception {
    return sha256(text.getBytes(StandardCharsets.UTF_8));
  }

  static String sha256(byte[] bytes) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
  }

  static String required(Map<String, String> options, String key) {
    String value = options.get(key);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + key + " is required");
    }
    return value;
  }
}
