description = "SemanticCache backed by a java-vectors VectorCollection"

dependencies {
    api(project(":vectors-cache"))
    api(project(":vectors-db"))

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testRuntimeOnly("ch.qos.logback:logback-classic:1.5.15")
}
