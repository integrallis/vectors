description = "Opt-in Azure SDK runtime for Azure Blob Storage as a Vectors storage backend"

// azure-sdk-bom 1.3.10 pins azure-storage-blob 12.37.2, azure-core 1.61.0, azure-core-http-jdk-httpclient 1.3.0.
val azureSdkBom = "1.3.10"

dependencies {
    api(project(":vectors-storage"))
    api(platform("com.azure:azure-sdk-bom:$azureSdkBom"))
    // The backend builds its clients on the JDK HttpClient; the SDK's default Netty transport (native
    // libraries, reactor-netty) is excluded so the runtime stays pure Java.
    api("com.azure:azure-storage-blob") {
        exclude(group = "com.azure", module = "azure-core-http-netty")
    }
    api("com.azure:azure-core-http-jdk-httpclient")
}
