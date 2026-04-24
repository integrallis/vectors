rootProject.name = "vectors"

include("vectors-core")
include("vectors-storage")
include("vectors-quantization")
include("vectors-hnsw")
include("vectors-vamana")
include("vectors-ivf")
include("vectors-db")
include("vectors-distributed")
include("vectors-gpu")
include("vectors-spring-ai")
include("vectors-spring-boot-starter")
include("vectors-langchain4j")
include("vectors-bench")
// --- VCR test harness modules ---
include("vectors-vcr-core")
include("vectors-vcr-semantic-db")
include("vectors-vcr-serde-avaje")
include("vectors-vcr-serde-jackson")
include("vectors-vcr-junit5")
include("vectors-vcr-testng")
include("vectors-vcr-spring-ai")
include("vectors-vcr-langchain4j")

// --- HTTP server (Helidon SE 4 / Níma) ---
include("vectors-server")

// --- Caching layer ---
include("vectors-cache")

include("docs")

// --- Runnable demos (not published) ---
include("demos:quickstart")
include("demos:spring-ai-rag")
include("demos:langchain4j-rag")
include("demos:embedding-cache")
include("demos:rerank-after-retrieval")
include("demos:vcr-e2e")

// Enable build cache
buildCache {
    local {
        isEnabled = true
    }
}
