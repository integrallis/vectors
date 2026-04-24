description = "Pluggable caching SPI (VectorCache / SemanticCache) with Caffeine default"

dependencies {
    api("com.github.ben-manes.caffeine:caffeine:3.1.8")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
