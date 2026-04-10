rootProject.name = "vectors"

include("vectors-core")
include("vectors-storage")
include("vectors-quantization")
include("vectors-hnsw")
include("vectors-vamana")
include("vectors-ivf")
include("vectors-db")
include("vectors-bench")
include("docs")

// Enable build cache
buildCache {
    local {
        isEnabled = true
    }
}
