description = "Optional GPU backend for java-vectors via Panama-FFM bindings to NVIDIA cuVS"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-storage"))

    // NVIDIA cuVS Panama-FFM bindings. Pulls a pure-Java JAR; the native `libcuvs.so` is
    // loaded at runtime only if a CUDA device is present. No JNI is maintained by us.
    // `api` scope so vectors-db can consume cuVS types (CagraIndex, BruteForceIndex, ...)
    // directly from its adapter implementations.
    api("com.nvidia.cuvs:cuvs-java:25.10.0")
}
