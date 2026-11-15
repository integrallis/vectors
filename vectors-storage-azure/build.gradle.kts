description = "Opt-in Azure SDK runtime for Azure Blob Storage as a Vectors storage backend"

// azure-sdk-bom 1.3.10 pins azure-storage-blob 12.37.2, azure-core 1.61.0, azure-core-http-jdk-httpclient 1.3.0.
val azureSdkBom = "1.3.10"
val jacksonVersion = rootProject.extra["jacksonVersion"] as String
val jacksonAnnotationsVersion = rootProject.extra["jacksonAnnotationsVersion"] as String
val slf4jVersion = rootProject.extra["slf4jVersion"] as String

dependencies {
    api(project(":vectors-storage"))
    api(platform("com.azure:azure-sdk-bom:$azureSdkBom"))
    // The backend builds its clients on the JDK HttpClient; the SDK's default Netty transport (native
    // libraries, reactor-netty) is excluded so the runtime stays pure Java.
    api("com.azure:azure-storage-blob") {
        exclude(group = "com.azure", module = "azure-core-http-netty")
    }
    api("com.azure:azure-core-http-jdk-httpclient")
    // azure-core declares Jackson 2.18 and SLF4J 1.7. Maven does not propagate this library's version
    // policy to the SDK's transitive dependencies (nearest declaration wins), so the policy-governed
    // versions are declared here directly, as vectors-storage-s3 does for its HTTP client line.
    listOf("jackson-core", "jackson-databind").forEach {
        runtimeOnly("com.fasterxml.jackson.core:$it:$jacksonVersion")
    }
    runtimeOnly("com.fasterxml.jackson.core:jackson-annotations:$jacksonAnnotationsVersion")
    runtimeOnly("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:$jacksonVersion")
    runtimeOnly("org.slf4j:slf4j-api:$slf4jVersion")
}
