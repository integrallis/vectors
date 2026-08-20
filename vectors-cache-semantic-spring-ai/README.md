# vectors-cache-semantic-spring-ai

Spring AI `ChatModel` decorator backed by a `SemanticCache`. It embeds the
complete prompt, reuses a sufficiently similar earlier answer, preserves the
model name, and marks cached responses so applications can report avoided model
calls accurately.

```java
ChatModel model = new SemanticCachingChatModel(delegate, embeddings, cache);
ChatResponse response = model.call(prompt);

if (SemanticCachingChatModel.isCacheHit(response)) {
    double similarity = SemanticCachingChatModel.cacheSimilarity(response).orElseThrow();
}
```

For a composed RAG prompt, mark the text that represents the user's semantic
intent. This keeps repeated instructions and retrieved context from creating
false matches:

```java
var message = UserMessage.builder()
    .text(completeRagPrompt)
    .metadata(Map.of(SemanticCachingChatModel.CACHE_KEY_METADATA, contextualUserQuery))
    .build();
var response = model.call(new Prompt(message));
```

Without that metadata, the decorator embeds the complete prompt for backwards
compatibility.

The cache must support entry attributes. The decorator uses them to prevent an
answer generated with one set of chat options from satisfying a request made
with different options. Responses containing tool calls and streaming responses
are not cached.

```kotlin
implementation("com.integrallis:vectors-cache-semantic-spring-ai:0.1.9")
implementation("com.integrallis:vectors-cache-semantic-db:0.1.9")
```
