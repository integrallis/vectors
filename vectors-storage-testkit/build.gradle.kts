description = "Storage backend test kit: the StorageBackend contract any backend implementation must pass"

dependencies {
    api(project(":vectors-storage"))
    // The contract is JUnit 5 default methods with AssertJ assertions; a backend's test class
    // implements the interface and supplies the backend under test.
    api("org.junit.jupiter:junit-jupiter-api:5.11.4")
    // JUnit's Gradle metadata keeps apiguardian (its @API markers) off the runtime graph while its
    // POM carries it; the published-graph gate requires the two to agree, and the annotations'
    // enum must stay on the compile classpath, so the library is declared at runtime here.
    runtimeOnly("org.apiguardian:apiguardian-api:1.1.2")
    api("org.assertj:assertj-core:3.27.2")
}
