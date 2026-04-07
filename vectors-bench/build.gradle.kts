plugins {
    id("me.champeau.jmh") version "0.7.2"
}

description = "JMH benchmarks and ANN-Benchmarks harness"

dependencies {
    implementation(project(":vectors-core"))
    implementation(project(":vectors-storage"))
    implementation(project(":vectors-quantization"))
    implementation(project(":vectors-hnsw"))
    implementation(project(":vectors-vamana"))
    implementation(project(":vectors-ivf"))

    implementation("org.openjdk.jmh:jmh-core:1.37")
    annotationProcessor("org.openjdk.jmh:jmh-generator-annprocess:1.37")
}

jmh {
    jvmArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}
