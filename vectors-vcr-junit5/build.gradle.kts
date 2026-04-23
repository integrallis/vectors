description = "JUnit 5 extension for VCR record/replay"

dependencies {
    api(project(":vectors-vcr-core"))
    compileOnly("org.junit.jupiter:junit-jupiter-api:5.11.4")
    testImplementation(project(":vectors-vcr-serde-avaje"))
    testImplementation("org.junit.platform:junit-platform-testkit:1.11.4")
}

// Scenario classes are driven via EngineTestKit from the real *Test classes; exclude from direct
// discovery so Gradle does not run them without the required system-property setup.
tasks.withType<Test>().configureEach {
    filter {
        excludeTestsMatching("*Scenario")
    }
}
