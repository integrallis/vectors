description = "Vectors Studio distributed backend — DistributedVectorCollection on R2 + sidecart text/blob"

dependencies {
    api(project(":vectors-studio-core"))
    api(project(":vectors-studio-sidecart"))
    api(project(":vectors-ivf"))
    api(project(":vectors-storage-s3"))

    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")

    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
