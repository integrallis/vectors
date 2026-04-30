plugins {
    application
}

description = "HTTP server for java-vectors (Helidon SE 4 / Nima, JSON-over-HTTP, port 8287)"

val helidonVersion = "4.3.4"
val jacksonVersion = "2.18.2"

dependencies {
    api(project(":vectors-db"))
    api(project(":vectors-core"))
    api(project(":vectors-hybrid"))
    runtimeOnly(project(":vectors-text-h2"))

    implementation(platform("io.helidon:helidon-dependencies:$helidonVersion"))
    implementation("io.helidon.webserver:helidon-webserver")
    implementation("io.helidon.webserver:helidon-webserver-sse")
    implementation("io.helidon.http.media:helidon-http-media-jackson")

    implementation(platform("com.fasterxml.jackson:jackson-bom:$jacksonVersion"))
    implementation("com.fasterxml.jackson.core:jackson-databind")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310")

    implementation("info.picocli:picocli:4.7.6")

    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.server.VectorsServer")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
