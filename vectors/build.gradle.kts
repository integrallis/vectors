// Umbrella dependency for the embedded java-vectors library. This module carries no code of its
// own; it exists so applications can depend on a single artifact — `com.integrallis:vectors` — that
// transitively brings in the whole embedded engine via `vectors-db` (core, storage, quantization,
// and the HNSW/Vamana/IVF index backends).
description = "Embedded java-vectors — single umbrella dependency (re-exports vectors-db)"

dependencies {
    api(project(":vectors-db"))
}

tasks.register("verifyRuntimeFootprint") {
    group = "verification"
    description = "Verify this facade stays lightweight and free of optional integrations"
    val facadeJar = tasks.named<Jar>("jar")
    dependsOn(configurations.runtimeClasspath, facadeJar)

    val maximumRuntimeBytes = 2L * 1024 * 1024
    val allowedExternalModules = setOf("org.slf4j:slf4j-api")

    doLast {
        val artifacts =
            configurations.runtimeClasspath.get().resolvedConfiguration.resolvedArtifacts
        val runtimeJars =
            (artifacts.map { it.file } + facadeJar.get().archiveFile.get().asFile).distinct()
        val unexpectedExternalModules =
            artifacts
                .map { "${it.moduleVersion.id.group}:${it.moduleVersion.id.name}" }
                .filter { !it.startsWith("com.integrallis:") && it !in allowedExternalModules }
                .distinct()
                .sorted()
        val runtimeBytes = runtimeJars.sumOf { it.length() }
        val failures = mutableListOf<String>()

        if (unexpectedExternalModules.isNotEmpty()) {
            failures +=
                "unexpected external modules: ${unexpectedExternalModules.joinToString()}"
        }
        if (runtimeBytes > maximumRuntimeBytes) {
            failures +=
                "runtime JARs total ${"%,d".format(runtimeBytes)} bytes; " +
                    "limit is ${"%,d".format(maximumRuntimeBytes)} bytes"
        }

        require(failures.isEmpty()) {
            "The com.integrallis:vectors facade must contain only the local engine and SLF4J:\n" +
                failures.joinToString("\n")
        }
        println(
            "  Facade runtime footprint: ${runtimeJars.size} JARs, " +
                "${"%,d".format(runtimeBytes)} bytes"
        )
    }
}
