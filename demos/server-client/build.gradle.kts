description = "End-to-end HTTP round-trip demo against an embedded vectors-server"

dependencies {
    implementation(project(":vectors-server"))
    implementation(project(":vectors-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.18.2")
    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")

    // CI-gating assertions on the demo's golden path (audit T3.10).
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("com.integrallis.vectors.demo.serverclient.ServerClientApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
