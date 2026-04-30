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
package com.integrallis.vectors.demo.rag.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.cache.CacheAdmissionPolicy;
import com.integrallis.vectors.cache.LLMResponseFilters;
import com.integrallis.vectors.demo.rag.model.CacheType;
import com.integrallis.vectors.demo.rag.model.ChatMessage;
import com.integrallis.vectors.demo.rag.model.LLMConfig;
import com.integrallis.vectors.demo.rag.model.Reference;
import com.integrallis.vectors.server.client.SearchHit;
import com.integrallis.vectors.server.client.VectorsServerClient;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * RAG (Retrieval-Augmented Generation) service using vectors-server for retrieval.
 *
 * <p>Orchestrates the RAG workflow:
 *
 * <ol>
 *   <li>Embed query and perform hybrid search via vectors-server
 *   <li>Build prompt with retrieved context (text + images)
 *   <li>Generate response with LLM
 *   <li>Track costs and tokens
 * </ol>
 */
public class RAGService {

  private final VectorsServerClient client;
  private final String collectionName;
  private final EmbeddingModel embeddingModel;
  private final ChatModel chatModel;
  private final CostTracker costTracker;
  private final LLMConfig config;

  private final CacheAdmissionPolicy<String> admissionPolicy = LLMResponseFilters.rejectRefusals();

  /** Semantic cache: stores (embedding, response, references) for cosine-similarity matching. */
  private final List<CacheEntry> semanticCache = new CopyOnWriteArrayList<>();

  private volatile double similarityThreshold = 0.75;

  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  /**
   * Tool: lists all images/charts available in the document with metadata. The LLM calls this when
   * the user wants to see what images exist (e.g., "list images", "what charts are in this
   * document?").
   */
  private static final ToolSpecification LIST_IMAGES_TOOL =
      ToolSpecification.builder()
          .name("list_images")
          .description(
              "Lists all images, charts, and figures available in the document. "
                  + "Returns their descriptions, page numbers, and dimensions. "
                  + "Use this when the user asks to list, enumerate, count, or see "
                  + "what images/charts/figures/visuals are in the document.")
          .build();

  /**
   * Tool: displays a specific image inline in the chat. The LLM calls this when the user wants to
   * see, view, or display a specific image. Accepts either a number or a text query.
   */
  private static final ToolSpecification DISPLAY_IMAGE_TOOL =
      ToolSpecification.builder()
          .name("display_image")
          .description(
              "Displays a specific image from the document inline in the chat. "
                  + "Use this when the user wants to see, view, show, or display an image. "
                  + "Provide EITHER image_number (from list_images) OR image_query "
                  + "(a description to search for, e.g. 'Financial Sonar pie chart'). "
                  + "Do NOT use this when the user asks to describe or analyze "
                  + "an image's content — for those, answer with text instead.")
          .parameters(
              JsonObjectSchema.builder()
                  .addIntegerProperty(
                      "image_number",
                      "The 1-based image number to display (matches the numbering "
                          + "from the list_images tool or image metadata list)")
                  .addStringProperty(
                      "image_query",
                      "A text description to search for (e.g. 'Financial Sonar', "
                          + "'revenue pie chart'). Used when the user refers to an image "
                          + "by description rather than number.")
                  .build())
          .build();

  private static final List<ToolSpecification> TOOLS =
      List.of(LIST_IMAGES_TOOL, DISPLAY_IMAGE_TOOL);

  private static final String SYSTEM_PROMPT =
      """
      You are a helpful assistant that answers questions based on the provided context \
      from a PDF document. The context includes text and image metadata.

      You have two tools available:
      - list_images: Call this when the user wants to list, enumerate, or count images/charts.
      - display_image: Call this when the user wants to see/view/display a specific image.

      Guidelines:
      - For text questions, answer directly from the text context.
      - For image listing questions, call the list_images tool.
      - For "show me image N" requests, call the display_image tool.
      - For questions ABOUT image content, answer from the image metadata in the context.
      - If the context doesn't contain relevant information, politely say so.
      - Always cite source pages when possible.
      """;

  /**
   * Creates a new RAGService.
   *
   * @param client vectors-server client for retrieval
   * @param collectionName collection to search
   * @param embeddingModel embedding model for query encoding
   * @param chatModel chat model for generation
   * @param costTracker cost tracker for calculating costs
   * @param config LLM configuration
   */
  public RAGService(
      VectorsServerClient client,
      String collectionName,
      EmbeddingModel embeddingModel,
      ChatModel chatModel,
      CostTracker costTracker,
      LLMConfig config) {
    this.client = client;
    this.collectionName = collectionName;
    this.embeddingModel = embeddingModel;
    this.chatModel = chatModel;
    this.costTracker = costTracker;
    this.config = config;
  }

  /**
   * Queries the RAG system.
   *
   * @param userQuery User's question
   * @param cacheType Type of caching to use (NONE or LOCAL)
   * @return Assistant response with cost tracking
   * @throws IllegalArgumentException if userQuery is null or empty
   */
  public ChatMessage query(String userQuery, CacheType cacheType) {
    if (userQuery == null || userQuery.trim().isEmpty()) {
      throw new IllegalArgumentException("User query cannot be null or empty");
    }

    // 1. Embed query (always needed — for semantic cache check and search)
    float[] queryVector = embeddingModel.embed(userQuery).content().vector();

    // 2. Check semantic cache if enabled
    if (cacheType == CacheType.LOCAL) {
      CacheEntry hit = findBestCacheMatch(queryVector);
      if (hit != null) {
        int tokenCount = costTracker.countTokens(hit.response());
        System.out.println("[CACHE HIT] Semantic cache hit for query: " + userQuery);
        return ChatMessage.assistant(
            hit.response(), tokenCount, 0.0, config.model(), true, hit.references());
      }
      System.out.println("[CACHE MISS] Semantic cache miss for query: " + userQuery);
    }

    // 3. Hybrid search — always text-only, never send image blobs to the LLM
    List<SearchHit> hits = client.hybridSearch(collectionName, queryVector, userQuery, 40, "RRF");
    System.out.println("Retrieved " + hits.size() + " results for query: " + userQuery);

    // Separate text and image hits
    List<SearchHit> textHits = new ArrayList<>();
    List<SearchHit> imageHits = new ArrayList<>();

    for (SearchHit hit : hits) {
      Map<String, Object> metadata = hit.metadata();
      String type =
          metadata != null ? String.valueOf(metadata.getOrDefault("type", "TEXT")) : "TEXT";
      if ("IMAGE".equals(type)) {
        imageHits.add(hit);
      } else if (!"PAGE_RENDER".equals(type)) {
        textHits.add(hit);
      }
    }

    List<SearchHit> allHits = new ArrayList<>(textHits);
    allHits.addAll(imageHits);
    List<Reference> references = extractReferences(allHits);

    // 4. Build text-only prompt with context + image metadata summary
    String contextText = buildContext(textHits);
    String imageMetadata = buildImageMetadataSummary(imageHits);

    StringBuilder prompt = new StringBuilder();
    if (!contextText.isEmpty()) {
      prompt.append("Text Context:\n").append(contextText).append("\n\n");
    }
    if (!imageMetadata.isEmpty()) {
      prompt.append(imageMetadata).append("\n\n");
    }
    prompt.append("Question: ").append(userQuery);

    List<dev.langchain4j.data.message.ChatMessage> messages = new ArrayList<>();
    messages.add(SystemMessage.from(SYSTEM_PROMPT));
    messages.add(UserMessage.from(prompt.toString()));

    // 5. Send to LLM with tool calling support — LLM decides when to use tools
    ChatRequest chatRequest =
        ChatRequest.builder().messages(messages).toolSpecifications(TOOLS).build();
    ChatResponse response = chatModel.chat(chatRequest);
    AiMessage aiMessage = response.aiMessage();

    // 6. Handle tool calls
    if (aiMessage.hasToolExecutionRequests()) {
      for (ToolExecutionRequest toolCall : aiMessage.toolExecutionRequests()) {
        String toolName = toolCall.name();
        System.out.println("[TOOL CALL] " + toolName + ": " + toolCall.arguments());
        if ("list_images".equals(toolName)) {
          return executeListImages();
        }
        if ("display_image".equals(toolName)) {
          return executeDisplayImage(toolCall);
        }
      }
    }

    String responseText = aiMessage.text();

    // 7. Store in semantic cache if enabled (skip error/refusal/clarification responses)
    if (cacheType == CacheType.LOCAL
        && responseText != null
        && admissionPolicy.test(responseText)) {
      semanticCache.add(new CacheEntry(queryVector.clone(), responseText, references));
      System.out.println("Cached response for query: " + userQuery);
    } else if (cacheType == CacheType.LOCAL) {
      System.out.println("Skipped caching (error/refusal response) for query: " + userQuery);
    }

    // 8. Calculate costs
    int tokenCount = costTracker.countTokens(responseText != null ? responseText : "");
    double cost = costTracker.calculateCost(config.provider(), config.model(), tokenCount);

    return ChatMessage.assistant(
        responseText != null ? responseText : "",
        tokenCount,
        cost,
        config.model(),
        false,
        references);
  }

  /** Extracts deduplicated references (by page) from search hits. */
  private List<Reference> extractReferences(List<SearchHit> hits) {
    List<Reference> references = new ArrayList<>();
    Set<Integer> seenPages = new HashSet<>();
    for (SearchHit hit : hits) {
      Map<String, Object> metadata = hit.metadata();
      Object pageObj = metadata != null ? metadata.get("page") : null;
      if (pageObj != null) {
        int page;
        if (pageObj instanceof Number num) {
          page = num.intValue();
        } else {
          page = Integer.parseInt(pageObj.toString());
        }
        if (seenPages.add(page)) {
          String type =
              metadata != null ? String.valueOf(metadata.getOrDefault("type", "TEXT")) : "TEXT";
          String preview = hit.text() != null ? hit.text() : "";
          references.add(Reference.of(page, type, preview, 80));
        }
      }
    }
    return references;
  }

  /**
   * Builds context string from text search hits.
   *
   * @param textHits text search results
   * @return Formatted context string
   */
  private String buildContext(List<SearchHit> textHits) {
    if (textHits == null || textHits.isEmpty()) {
      return "";
    }

    StringBuilder sb = new StringBuilder();
    for (SearchHit hit : textHits) {
      if (hit.text() != null && !hit.text().isEmpty()) {
        if (sb.length() > 0) {
          sb.append("\n\n");
        }
        sb.append(hit.text());
      }
    }
    return sb.toString();
  }

  /**
   * Builds a brief image metadata summary to include in the text context. This tells the LLM what
   * images are available without sending any actual image data (blobs).
   */
  private static String buildImageMetadataSummary(List<SearchHit> imageHits) {
    if (imageHits.isEmpty()) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    sb.append("Image Metadata (").append(imageHits.size()).append(" images available");
    sb.append(" — use list_images tool for full list, display_image tool to show one):\n");
    int limit = Math.min(10, imageHits.size());
    for (int i = 0; i < limit; i++) {
      SearchHit hit = imageHits.get(i);
      sb.append("  [Image ").append(i + 1).append("] ");
      sb.append(hit.text() != null ? hit.text() : "unknown");
      sb.append("\n");
    }
    if (imageHits.size() > limit) {
      sb.append("  ... and ").append(imageHits.size() - limit).append(" more\n");
    }
    return sb.toString();
  }

  /**
   * Sets the cosine similarity threshold for semantic cache matching.
   *
   * @param threshold value between 0.0 and 1.0 (default 0.9)
   */
  public void setSimilarityThreshold(double threshold) {
    this.similarityThreshold = threshold;
  }

  /** Finds the best-matching cache entry whose cosine similarity meets the threshold. */
  private CacheEntry findBestCacheMatch(float[] queryVector) {
    double bestSim = -1.0;
    CacheEntry bestEntry = null;
    for (CacheEntry entry : semanticCache) {
      float sim = cosineSimilarity(queryVector, entry.queryVector());
      if (sim >= similarityThreshold && sim > bestSim) {
        bestSim = sim;
        bestEntry = entry;
      }
    }
    return bestEntry;
  }

  /** Computes cosine similarity between two vectors. */
  static float cosineSimilarity(float[] a, float[] b) {
    float dot = 0f, normA = 0f, normB = 0f;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
    return denom == 0f ? 0f : dot / denom;
  }

  record CacheEntry(float[] queryVector, String response, List<Reference> references) {}

  /** Clears all cached responses. Should be called when the document context changes. */
  public void clearCache() {
    semanticCache.clear();
  }

  /**
   * Executes the list_images tool call. Fetches all IMAGE chunks from vectors-server, filters to
   * only those with available blobs (consistent with display_image numbering), and returns a
   * deterministic, formatted listing. No LLM involvement — no hallucination risk.
   */
  private ChatMessage executeListImages() {
    try {
      List<SearchHit> available = fetchDisplayableImages();

      if (available.isEmpty()) {
        return ChatMessage.assistant(
            "No images were found in this document.", 0, 0.0, config.model(), false);
      }

      StringBuilder sb = new StringBuilder();
      sb.append("The document contains ").append(available.size()).append(" image(s):\n\n");
      for (int i = 0; i < available.size(); i++) {
        SearchHit hit = available.get(i);
        sb.append(i + 1).append(". ");
        if (hit.text() != null && !hit.text().isEmpty()) {
          sb.append(hit.text());
        } else {
          Map<String, Object> meta = hit.metadata();
          if (meta != null) {
            Object page = meta.get("page");
            sb.append("Image on page ").append(page != null ? page : "unknown");
          } else {
            sb.append("Image (metadata unavailable)");
          }
        }
        sb.append("\n");
      }
      sb.append("\nTo display an image, say \"display image N\" where N is the image number.");

      String listing = sb.toString().trim();
      int tokenCount = costTracker.countTokens(listing);
      List<Reference> references = extractReferences(available);
      return ChatMessage.assistant(listing, tokenCount, 0.0, config.model(), false, references);

    } catch (Exception e) {
      return ChatMessage.assistant(
          "Failed to list images: " + e.getMessage(), 0, 0.0, config.model(), false);
    }
  }

  /**
   * Executes the display_image tool call by fetching the requested image blob from vectors-server.
   *
   * @param toolCall the LLM's tool execution request containing the image_number argument
   * @return a ChatMessage with inline image data, or an error message if the image is not found
   */
  private ChatMessage executeDisplayImage(ToolExecutionRequest toolCall) {
    try {
      List<SearchHit> available = fetchDisplayableImages();

      if (available.isEmpty()) {
        return ChatMessage.assistant(
            "No images are available in this document.", 0, 0.0, config.model(), false);
      }

      // Resolve image number — either from direct number or text search
      int imageNumber = resolveImageNumber(toolCall.arguments(), available);

      if (imageNumber < 1 || imageNumber > available.size()) {
        return ChatMessage.assistant(
            "Image not found. There are "
                + available.size()
                + " images available. Use \"list images\" to see them.",
            0,
            0.0,
            config.model(),
            false);
      }

      SearchHit target = available.get(imageNumber - 1);
      Optional<byte[]> blobOpt = client.getBlob(collectionName, target.id());
      if (blobOpt.isEmpty()) {
        return ChatMessage.assistant(
            "Image " + imageNumber + " data is not available.", 0, 0.0, config.model(), false);
      }

      String caption = "Image " + imageNumber + ": " + (target.text() != null ? target.text() : "");
      return ChatMessage.imageMessage(caption, blobOpt.get(), 0);

    } catch (Exception e) {
      return ChatMessage.assistant(
          "Failed to display image: " + e.getMessage(), 0, 0.0, config.model(), false);
    }
  }

  /**
   * Resolves the image number from tool call arguments. If image_number is given, uses it directly.
   * If image_query is given, searches available images by text match and returns the best match.
   */
  private static int resolveImageNumber(String arguments, List<SearchHit> available) {
    try {
      JsonNode node = OBJECT_MAPPER.readTree(arguments);

      // Prefer image_number if present
      JsonNode numberNode = node.get("image_number");
      if (numberNode != null && !numberNode.isNull() && numberNode.asInt() > 0) {
        return numberNode.asInt();
      }

      // Fall back to image_query text search
      JsonNode queryNode = node.get("image_query");
      if (queryNode != null && !queryNode.isNull()) {
        String query = queryNode.asText().toLowerCase();
        int bestIndex = -1;
        int bestScore = 0;
        String[] queryWords = query.split("\\s+");
        for (int i = 0; i < available.size(); i++) {
          String text = available.get(i).text();
          if (text == null) continue;
          String lowerText = text.toLowerCase();
          int score = 0;
          if (lowerText.contains(query)) score += 100;
          for (String word : queryWords) {
            if (word.length() >= 3 && lowerText.contains(word)) score += 10;
          }
          if (score > bestScore) {
            bestScore = score;
            bestIndex = i + 1;
          }
        }
        return bestScore >= 10 ? bestIndex : -1;
      }

      return -1;
    } catch (Exception e) {
      throw new IllegalArgumentException("Failed to parse tool arguments: " + arguments, e);
    }
  }

  /**
   * Fetches all IMAGE chunks that have available blobs, sorted by page number then ID. Used by both
   * list_images and display_image tools to ensure consistent numbering.
   */
  private List<SearchHit> fetchDisplayableImages() {
    float[] queryVector = embeddingModel.embed("images").content().vector();
    List<SearchHit> imageHits =
        new ArrayList<>(
            client.search(
                collectionName, queryVector, 200, null, Map.of("field", "type", "eq", "IMAGE")));

    // Sort by page number then ID for deterministic ordering
    imageHits.sort(
        (a, b) -> {
          int pageA = extractPageNumber(a);
          int pageB = extractPageNumber(b);
          if (pageA != pageB) return Integer.compare(pageA, pageB);
          return a.id().compareTo(b.id());
        });

    // Filter to only images with available blobs
    List<SearchHit> displayable = new ArrayList<>();
    for (SearchHit hit : imageHits) {
      try {
        Optional<byte[]> blobOpt = client.getBlob(collectionName, hit.id());
        if (blobOpt.isPresent() && blobOpt.get().length > 0) {
          displayable.add(hit);
        }
      } catch (Exception e) {
        // Skip images whose blobs can't be fetched
      }
    }
    return displayable;
  }

  /**
   * Parses the image_number from a tool call's JSON arguments.
   *
   * @param arguments JSON string like {@code {"image_number": 5}}
   * @return the parsed image number
   * @throws IllegalArgumentException if the arguments cannot be parsed
   */
  private static int parseImageNumber(String arguments) {
    try {
      JsonNode node = OBJECT_MAPPER.readTree(arguments);
      JsonNode imageNumber = node.get("image_number");
      if (imageNumber != null) {
        return imageNumber.asInt();
      }
      throw new IllegalArgumentException("Missing 'image_number' in tool arguments: " + arguments);
    } catch (Exception e) {
      throw new IllegalArgumentException("Failed to parse tool arguments: " + arguments, e);
    }
  }

  /** Extracts the page number from a search hit's metadata. Returns 0 if not available. */
  private static int extractPageNumber(SearchHit hit) {
    if (hit.metadata() == null) return 0;
    Object pageObj = hit.metadata().get("page");
    if (pageObj instanceof Number num) return num.intValue();
    if (pageObj != null) {
      try {
        return Integer.parseInt(pageObj.toString());
      } catch (NumberFormatException e) {
        return 0;
      }
    }
    return 0;
  }
}
