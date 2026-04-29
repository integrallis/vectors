plugins {
    application
}

description = "Vectors Studio — Helidon SE 4 web frontend (HTMX + JTE + Three.js projector island, port 8288)"

val helidonVersion = "4.3.4"
val jacksonVersion = "2.18.2"
val jteVersion = "3.2.3"
val langchain4jVersion = "1.13.1"

dependencies {
    implementation(project(":vectors-studio-core"))
    implementation(project(":vectors-db"))
    implementation(project(":vectors-server-client"))

    implementation(platform("io.helidon:helidon-dependencies:$helidonVersion"))
    implementation("io.helidon.webserver:helidon-webserver")
    implementation("io.helidon.webserver:helidon-webserver-sse")
    implementation("io.helidon.webserver:helidon-webserver-static-content")
    implementation("io.helidon.http.media:helidon-http-media-jackson")

    implementation(platform("com.fasterxml.jackson:jackson-bom:$jacksonVersion"))
    implementation("com.fasterxml.jackson.core:jackson-databind")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310")

    implementation("gg.jte:jte:$jteVersion")
    implementation("gg.jte:jte-runtime:$jteVersion")

    implementation("info.picocli:picocli:4.7.6")

    compileOnly("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j-core:$langchain4jVersion")

    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.mockito:mockito-core:5.14.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.studio.web.StudioServer")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}
