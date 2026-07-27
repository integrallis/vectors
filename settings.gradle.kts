pluginManagement {
    repositories {
        gradlePluginPortal()
        // The first-party MFCQI plugin (com.integrallis.mfcqi) is published to Maven Central,
        // not the Gradle Plugin Portal.
        mavenCentral()
    }
}

rootProject.name = "vectors"

include("vectors-core")
include("vectors-storage")
include("vectors-quantization")
include("vectors-hnsw")
include("vectors-vamana")
include("vectors-ivf")
include("vectors-ingest")
include("vectors-hybrid")
include("vectors-router")
include("vectors-optimizer")
include("vectors-text-h2")
include("vectors-db")
// Umbrella artifact: publishes as `com.integrallis:vectors`, re-exporting vectors-db.
include("vectors")
include("vectors-distributed")
include("vectors-cluster")
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
include("vectors-server-client")

// --- Caching layer ---
include("vectors-cache")
include("vectors-cache-jcache")
include("vectors-cache-semantic-db")
include("vectors-cache-spring-ai")
include("vectors-cache-langchain4j")

// --- Studio (exploration & DR workbench) ---
include("vectors-studio-core")
include("vectors-studio-sidecart")
include("vectors-studio-distributed")
include("vectors-studio-web")

include("docs")

// --- Runnable demos (not published) ---
include("demos:quickstart")
include("demos:spring-ai-rag")
include("demos:langchain4j-rag")
include("demos:embedding-cache")
include("demos:rerank-after-retrieval")
include("demos:vcr-e2e")
include("demos:server-client")
include("demos:semantic-cache")
include("demos:rag-multimodal")
include("demos:studio-r2-sidecart")
include("demos:optimizer-tutorial")

// Enable build cache
buildCache {
    local {
        isEnabled = true
    }
}
