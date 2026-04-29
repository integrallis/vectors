description = "Vectors Studio — UI-agnostic domain core (backends, projection SPI, recommender)"

val smileVersion = "6.0.0"
val langchain4jVersion = "1.13.1"

dependencies {
    api(project(":vectors-core"))
    api(project(":vectors-db"))
    api(project(":vectors-server-client"))

    implementation("com.github.haifengl:smile-core:$smileVersion")

    compileOnly("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j-core:$langchain4jVersion")

    testImplementation(project(":vectors-server"))

    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.mockito:mockito-core:5.14.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
