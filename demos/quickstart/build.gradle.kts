description = "Quickstart demo: open a VectorCollection, add vectors, search"

dependencies {
    implementation(project(":vectors-db"))

    // CI-gating assertions on the demo's golden path (audit T3.10). Versions kept in sync with
    // the central library testImplementation block in the root build.gradle.kts.
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.demo.quickstart.QuickstartApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
