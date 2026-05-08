description = "Vectors Studio sidecart sources — pluggable text/blob backends (file, H2, HTTP) keyed by document id"

dependencies {
    api(project(":vectors-studio-core"))
    api(project(":vectors-ingest"))

    implementation("com.h2database:h2:2.3.232")

    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
