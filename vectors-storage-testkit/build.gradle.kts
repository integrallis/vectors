description = "Storage backend test kit: the StorageBackend contract any backend implementation must pass"

dependencies {
    api(project(":vectors-storage"))
    // The contract is JUnit 5 default methods with AssertJ assertions; a backend's test class
    // implements the interface and supplies the backend under test.
    api("org.junit.jupiter:junit-jupiter-api:5.11.4")
    api("org.assertj:assertj-core:3.27.2")
}
