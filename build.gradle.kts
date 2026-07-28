import groovy.json.JsonSlurper
import java.security.MessageDigest

plugins {
    java
    `java-library`
    `maven-publish`
    id("com.github.spotbugs") version "6.4.4" apply false
    id("com.diffplug.spotless") version "6.25.0" apply false
    id("org.cyclonedx.bom") version "3.2.4" apply false
    id("org.owasp.dependencycheck") version "12.2.1" apply false
    id("com.integrallis.mfcqi") version "0.7.0"
    jacoco
}

val notebookRepositoryUrl = providers.gradleProperty("notebookRepository")
    .orElse(providers.environmentVariable("VECTORS_NOTEBOOK_REPOSITORY"))
    .map { it.trim() }
    .filter { it.isNotEmpty() }

allprojects {
    group = "com.integrallis"

    repositories {
        if (project == rootProject && notebookRepositoryUrl.isPresent) {
            maven {
                name = "notebookCandidate"
                url = uri(notebookRepositoryUrl.get())
                content {
                    includeGroup("com.integrallis")
                }
            }
        }
        mavenCentral()
    }
}

// Library subprojects (excludes docs and runnable demos)
val libraryProjects = subprojects.filter {
    it.name != "docs" && it.path != ":demos" && !it.path.startsWith(":demos:")
}
val demoProjects = subprojects.filter { it.path == ":demos" || it.path.startsWith(":demos:") }

// FSL-1.1-ALv2 modules — all others are Apache 2.0
val fslModules = setOf("vectors-distributed", "vectors-cluster", "vectors-server", "vectors-gpu")

// Maven Central 0.1.x scope. This is an allowlist so a new or experimental module can never become
// public merely by being added to settings.gradle.kts.
val publishedModuleNames = setOf(
    "vectors-core",
    "vectors-storage",
    "vectors-storage-s3",
    "vectors-quantization",
    "vectors-hnsw",
    "vectors-vamana",
    "vectors-ivf",
    "vectors-db",
    "vectors-db-arrow",
    "vectors",
    "vectors-hybrid",
    "vectors-spring-ai",
    "vectors-langchain4j",
    "vectors-spring-boot-starter",
    "vectors-cache",
    "vectors-cache-jcache",
    "vectors-cache-langchain4j",
    "vectors-cache-semantic-db",
    "vectors-cache-spring-ai",
    "vectors-vcr-core",
    "vectors-vcr-semantic-db",
    "vectors-vcr-serde-avaje",
    "vectors-vcr-serde-jackson",
    "vectors-vcr-junit5",
    "vectors-vcr-testng",
    "vectors-vcr-spring-ai",
    "vectors-vcr-langchain4j"
)
extra["publishedModuleNames"] = publishedModuleNames
extra["publishedJavadocModuleNames"] =
    publishedModuleNames - setOf("vectors", "vectors-storage-s3", "vectors-db-arrow")
val publishedProjects = libraryProjects.filter { it.name in publishedModuleNames }

// JJava notebooks consume one generated classpath so notebook cells never encode module JAR names
// or versions. Source mode resolves project runtime variants; release mode resolves the exact
// Maven Central artifacts selected with -PnotebookVersion (or VECTORS_VERSION).
val notebookMode = providers.gradleProperty("notebookMode")
    .orElse(providers.environmentVariable("VECTORS_NOTEBOOK_MODE"))
    .orElse("source")
val notebookVersion = providers.gradleProperty("notebookVersion")
    .orElse(providers.environmentVariable("VECTORS_VERSION"))
    .orElse(provider { project.version.toString() })

fun Configuration.asNotebookRuntimeClasspath() {
    isCanBeConsumed = false
    isCanBeResolved = true
    attributes {
        attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
    }
}

val notebookSourceClasspath by configurations.creating {
    description = "Runtime classpath for notebooks using the current project sources"
    asNotebookRuntimeClasspath()
}
val notebookReleaseClasspath by configurations.creating {
    description = "Runtime classpath for notebooks using released com.integrallis artifacts"
    asNotebookRuntimeClasspath()
}

dependencies {
    listOf(
        ":vectors",
        ":vectors-spring-ai",
        ":vectors-langchain4j",
        ":vectors-cache-langchain4j",
        ":vectors-vcr-serde-jackson"
    ).forEach { notebookSourceClasspath(project(it)) }

    listOf(
        "vectors",
        "vectors-spring-ai",
        "vectors-langchain4j",
        "vectors-cache-langchain4j",
        "vectors-vcr-serde-jackson"
    ).forEach { artifact ->
        notebookReleaseClasspath("com.integrallis:$artifact:${notebookVersion.get()}")
    }

    listOf(
        "org.springframework.ai:spring-ai-vector-store:1.1.4",
        "org.springframework.ai:spring-ai-model:1.1.4",
        "io.micrometer:micrometer-observation:1.14.4",
        "dev.langchain4j:langchain4j-core:1.13.1",
        "org.slf4j:slf4j-nop:2.0.17"
    ).forEach { dependency ->
        notebookSourceClasspath(dependency)
        notebookReleaseClasspath(dependency)
    }
}

// ---------------------------------------------------------------------------
// MFCQI (Multi-Factor Code Quality Index) — https://github.com/integrallis/mfcqi-java
// The root task produces the aggregate badge over the published-library production sources (the
// shipped-product signal); each library module additionally scores its own sources for a per-module
// badge. Pure source-tree analysis (bytecodeSecurity off) — no compiled classpath needed.
//
// MFCQI excludes build/ output dirs, so the aggregate cannot be staged there. The MFCQI workflow
// stages the published sources into a temp dir outside the repo and passes -Pmfcqi.sourceDir; the
// root task reads that (falling back to the repo root for a quick local run).
// ---------------------------------------------------------------------------
mfcqi {
    source.set(
        layout.projectDirectory.dir(providers.gradleProperty("mfcqi.sourceDir").getOrElse("."))
    )
    bytecodeSecurity.set(false)
    failOnGate.set(false)
}

// Per-module MFCQI: every library module with production sources gets its own score + badge JSON.
configure(libraryProjects.filter { it.file("src/main/java").isDirectory }) {
    apply(plugin = "com.integrallis.mfcqi")
    configure<com.integrallis.mfcqi.gradle.MfcqiExtension> {
        source.set(layout.projectDirectory.dir("src/main/java"))
        bytecodeSecurity.set(false)
        failOnGate.set(false)
    }
}

val apacheLicenseHeader = """
    /*
     * Copyright 2025-2026 Integrallis Software, LLC
     *
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     *     https://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     */
""".trimIndent()

val fslLicenseHeader = """
    /*
     * Copyright 2025-2026 Integrallis Software, LLC
     *
     * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
     * (the "License"); you may not use this file except in compliance with the License.
     *
     *     https://fsl.software/FSL-1.1-ALv2.txt
     *
     * Unless required by applicable law or agreed to in writing, software distributed under
     * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
     * ANY KIND, either express or implied. See the License for the specific language
     * governing permissions and limitations under the License.
     *
     * Change Date: April 25, 2028
     * Change License: Apache License, Version 2.0
     */
""".trimIndent()

configure(libraryProjects) {
    apply(plugin = "java")
    apply(plugin = "java-library")
    apply(plugin = "com.github.spotbugs")
    apply(plugin = "com.diffplug.spotless")
    apply(plugin = "jacoco")
    apply(plugin = "org.owasp.dependencycheck")
    if (project.name in publishedModuleNames) {
        apply(plugin = "org.cyclonedx.bom")
        // CycloneDX 3.2 registers an outgoing configuration from this task lazily. Realize it
        // before maven-publish observes the Java variants; otherwise Gradle 9 rejects the plugin's
        // later attempt to mutate an already-consumed configuration.
        tasks.named("cyclonedxDirectBom").get()
    }

    // Dependency locking — enforced when lockfiles exist, lenient otherwise
    dependencyLocking {
        lockAllConfigurations()
    }

    tasks.register("resolveAndLockAllConfigurations") {
        group = "verification"
        description = "Resolve this module's dependencies and write its lockfile"
        notCompatibleWithConfigurationCache("Resolves and locks every module configuration")
        doFirst {
            require(gradle.startParameter.isWriteDependencyLocks) {
                "${path} must be run with the --write-locks flag"
            }
        }
        doLast {
            configurations.filter { it.isCanBeResolved }.forEach { configuration ->
                configuration.resolve()
            }
        }
    }

    java {
        toolchain {
            languageVersion = JavaLanguageVersion.of(25)
        }
        withJavadocJar()
        withSourcesJar()
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
        options.compilerArgs.addAll(listOf(
            "-parameters",
            "-Xlint:all",
            "-Xlint:-processing",
            "-Xlint:-incubating",
            "-Xlint:-classfile",   // suppress classfile annotation warnings from third-party JARs
            "-Werror",
            "--add-modules", "jdk.incubator.vector"
        ))
    }

    // Common JVM args and logging for ALL Test tasks — no tag filters here.
    // Tag filters must live on each individual task so they do not accumulate
    // (tasks.withType<Test> applies to every Test task including slowTest and
    // unitTest; adding excludeTags here would fight with their includeTags).
    tasks.withType<Test> {
        jvmArgs("--add-modules", "jdk.incubator.vector")
        testLogging {
            events("passed", "skipped", "failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            showStandardStreams = false
        }
        // Per-module test fork count. With org.gradle.parallel=true several modules' test
        // tasks run concurrently, so each module forking availableProcessors() JVMs would
        // wildly oversubscribe the host (N modules x 32 forks) and starve timing-sensitive
        // concurrency tests. A quarter of the cores per module keeps the total near the
        // core count while still parallelising within a module.
        maxParallelForks = (Runtime.getRuntime().availableProcessors() / 4).coerceAtLeast(1)
    }

    // Custom test tasks — must wire testClassesDirs and classpath so Gradle finds compiled tests
    tasks.register<Test>("unitTest") {
        description = "Run only unit tests"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("unit")
        }
        testLogging {
            events("passed", "skipped", "failed")
        }
    }

    tasks.register<Test>("slowTest") {
        description = "Run slow tests (large datasets, extended scenarios)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("slow")
        }
        filter { isFailOnNoMatchingTests = false }
        testLogging {
            events("passed", "skipped", "failed")
        }
    }

    // Integration tests: require a Docker daemon (TestContainers + LocalStack).
    // Run with: ./gradlew integrationTest
    // Excluded from the default 'test' task to avoid breaking CI without Docker.
    tasks.register<Test>("integrationTest") {
        description = "Run integration tests that require Docker (TestContainers + LocalStack)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("integration")
        }
        // Modules with no @Tag("integration") tests are a clean no-op rather than a task failure.
        filter { isFailOnNoMatchingTests = false }
        testLogging {
            events("passed", "skipped", "failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        }
        // Prevent parallel forks inside an integration test run: containers are shared
        // at the class level (@Container static field) but parallel class execution can
        // race on bucket creation / deletion. Sequential execution per module is safer.
        maxParallelForks = 1
    }

    // Distributed tests: multi-node clusters over real HTTP transport (Docker required).
    tasks.register<Test>("distributedTest") {
        description = "Run distributed cluster tests (requires Docker)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("distributed")
        }
        filter { isFailOnNoMatchingTests = false }
        maxParallelForks = 1
    }

    // Kubernetes deployment tests: K3s via TestContainers (Docker required).
    tasks.register<Test>("k8sTest") {
        description = "Run Kubernetes deployment tests (requires Docker)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("k8s")
        }
        filter { isFailOnNoMatchingTests = false }
        maxParallelForks = 1
    }

    // Chaos engineering tests: fault injection during operations (Docker required).
    tasks.register<Test>("chaosTest") {
        description = "Run chaos engineering tests (requires Docker)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("chaos")
        }
        filter { isFailOnNoMatchingTests = false }
        maxParallelForks = 1
    }

    // Large-scale scalability tests (Docker required, long-running).
    tasks.register<Test>("scaleTest") {
        description = "Run scalability tests (requires Docker, long-running)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("scale")
        }
        filter { isFailOnNoMatchingTests = false }
        maxParallelForks = 1
        systemProperty("junit.jupiter.execution.timeout.default", "30m")
    }

    // Recall regression gate (P1.1): deterministic recall@10 vs a committed baseline, no downloads.
    // Run via ./gradlew :vectors-bench:recallGate (wired into CI). Record a fresh baseline with
    // -Drecall.gate.mode=record. Lives only in vectors-bench; other modules' task is a no-op.
    tasks.register<Test>("recallGate") {
        description = "Recall@10 regression gate vs committed baseline (deterministic, no downloads)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("recall")
        }
        // Lives only in vectors-bench; other modules' task is a documented no-op.
        filter { isFailOnNoMatchingTests = false }
        // One heavy, deterministic test class — no intra-task forking, and forward the mode flag.
        maxParallelForks = 1
        systemProperty("recall.gate.mode", System.getProperty("recall.gate.mode", "verify"))
        testLogging {
            events("passed", "skipped", "failed")
            showStandardStreams = true
        }
    }

    // Default 'test' task excludes all infrastructure-heavy tags.
    tasks.named<Test>("test") {
        useJUnitPlatform {
            excludeTags(
                "slow", "benchmark", "integration", "distributed", "k8s", "chaos", "scale", "recall")
        }
    }

    tasks.withType<Javadoc> {
        val javadocOptions = options as StandardJavadocDocletOptions
        javadocOptions.addBooleanOption("Xdoclint:all,-missing", true)
        javadocOptions.addBooleanOption("html5", true)
        javadocOptions.addStringOption("-add-modules", "jdk.incubator.vector")
        isFailOnError = true
    }

    configure<com.diffplug.gradle.spotless.SpotlessExtension> {
        java {
            googleJavaFormat("1.35.0")
            removeUnusedImports()
            trimTrailingWhitespace()
            endWithNewline()
            licenseHeader(if (project.name in fslModules) fslLicenseHeader else apacheLicenseHeader)
        }
    }

    // Configure SpotBugs
    tasks.withType<com.github.spotbugs.snom.SpotBugsTask> {
        excludeFilter.set(file("${rootProject.projectDir}/spotbugs-exclude.xml"))
    }

    // Disable SpotBugs for test code
    tasks.named("spotbugsTest") {
        enabled = false
    }

    tasks.jacocoTestReport {
        dependsOn(tasks.test)
        reports {
            xml.required = true
            html.required = true
        }
    }

    tasks.jacocoTestCoverageVerification {
        dependsOn(tasks.test)
        enabled = project in publishedProjects
        violationRules {
            rule {
                limit {
                    minimum = "0.80".toBigDecimal()
                }
            }
        }
    }

    if (project in publishedProjects) {
        tasks.named("check") {
            dependsOn(tasks.jacocoTestCoverageVerification)
        }
    }

    // OWASP Dependency-Check — runs only when explicitly invoked
    configure<org.owasp.dependencycheck.gradle.extension.DependencyCheckExtension> {
        failBuildOnCVSS = 7.0f
        formats.set(listOf("HTML", "JSON", "SARIF"))
        outputDirectory.set(layout.buildDirectory.dir("reports/dependency-check"))
        suppressionFile = "${rootProject.projectDir}/owasp-suppressions.xml"
        nvd.apiKey = System.getenv("NVD_API_KEY") ?: ""
        analyzers.ossIndex.enabled = true
        analyzers.ossIndex.username = System.getenv("OSS_INDEX_USERNAME") ?: ""
        analyzers.ossIndex.password = System.getenv("OSS_INDEX_TOKEN") ?: ""
    }

    // Reproducible JAR manifest attributes
    tasks.withType<Jar>().configureEach {
        manifest {
            attributes(
                "Build-Jdk-Spec" to "25",
                "Created-By" to "Gradle ${gradle.gradleVersion}"
            )
        }
    }

    dependencies {
        // Logging
        implementation("org.slf4j:slf4j-api:2.0.17")

        // Testing
        testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
        testImplementation("org.assertj:assertj-core:3.27.2")
        testImplementation("org.mockito:mockito-core:5.15.2")
        testImplementation("org.mockito:mockito-junit-jupiter:5.15.2")
        testRuntimeOnly("org.junit.platform:junit-platform-launcher")
        testRuntimeOnly("ch.qos.logback:logback-classic:1.5.15")
    }
}

// ---------------------------------------------------------------------------
// Publishing: stage the Apache-licensed 0.1.x library set for JReleaser.
// ---------------------------------------------------------------------------

configure(publishedProjects) {
    apply(plugin = "maven-publish")

    configure<PublishingExtension> {
        publications {
            create<MavenPublication>("maven") {
                from(components["java"])
                pom {
                    name.set(project.name)
                    description.set(provider { project.description ?: "java-vectors — ${project.name}" })
                    url.set("https://github.com/integrallis/vectors")
                    licenses {
                        license {
                            name.set("The Apache License, Version 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                            distribution.set("repo")
                        }
                    }
                    developers {
                        developer {
                            id.set("bsbodden")
                            name.set("Brian Sam-Bodden")
                            email.set("bsbodden@gmail.com")
                            organization.set("Integrallis Software")
                            organizationUrl.set("https://integrallis.com")
                        }
                    }
                    scm {
                        connection.set("scm:git:https://github.com/integrallis/vectors.git")
                        developerConnection.set("scm:git:ssh://git@github.com/integrallis/vectors.git")
                        url.set("https://github.com/integrallis/vectors")
                    }
                }
            }
        }
        repositories {
            maven {
                name = "staging"
                url = uri(rootProject.layout.buildDirectory.dir("staging-deploy").get().asFile)
            }
        }
    }
}

// Runnable demo subprojects — same toolchain and Vector API flags as the libraries,
// but not published as Maven artifacts and not subject to the library coverage gate.
configure(demoProjects) {
    apply(plugin = "java")
    apply(plugin = "application")
    apply(plugin = "com.diffplug.spotless")

    extensions.configure<JavaPluginExtension> {
        toolchain {
            languageVersion = JavaLanguageVersion.of(25)
        }
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
        options.compilerArgs.addAll(listOf(
            "-parameters",
            "-Xlint:all",
            "-Xlint:-processing",
            "-Xlint:-incubating",
            "-Xlint:-classfile",
            "-Werror",
            "--add-modules", "jdk.incubator.vector"
        ))
    }

    tasks.withType<JavaExec> {
        jvmArgs("--add-modules", "jdk.incubator.vector")
    }

    tasks.withType<Test> {
        useJUnitPlatform()
        jvmArgs("--add-modules", "jdk.incubator.vector")
    }

    configure<com.diffplug.gradle.spotless.SpotlessExtension> {
        java {
            googleJavaFormat("1.35.0")
            removeUnusedImports()
            trimTrailingWhitespace()
            endWithNewline()
            licenseHeader(apacheLicenseHeader)
        }
    }

    dependencies {
        "implementation"("org.slf4j:slf4j-api:2.0.16")
        "runtimeOnly"("ch.qos.logback:logback-classic:1.5.15")
    }
}

// ---------------------------------------------------------------------------
// Compliance verification tasks
// ---------------------------------------------------------------------------

tasks.register("verifySbom") {
    group = "verification"
    description = "Verify CycloneDX SBOM generation for every published module"
    dependsOn(publishedProjects.map { "${it.path}:cyclonedxDirectBom" })
    doLast {
        publishedProjects.forEach { proj ->
            val file = proj.layout.buildDirectory.file("reports/cyclonedx-direct/bom.json").get().asFile
            require(file.exists()) { "SBOM not found: ${file.absolutePath}" }
            @Suppress("UNCHECKED_CAST")
            val json = groovy.json.JsonSlurper().parseText(file.readText()) as Map<String, Any?>
            require(json["bomFormat"] == "CycloneDX") {
                "Invalid bomFormat in ${proj.name}: ${json["bomFormat"]}"
            }
            val specVersion = json["specVersion"] as? String
            require(specVersion != null && specVersion.startsWith("1.")) {
                "Invalid specVersion in ${proj.name}: $specVersion"
            }
            @Suppress("UNCHECKED_CAST")
            val metadata = json["metadata"] as? Map<String, Any?>
            require(metadata != null) { "Missing metadata in ${proj.name}" }
            println("  SBOM valid: ${proj.name} (CycloneDX $specVersion)")
        }
    }
}

tasks.register("verifyGovernanceFiles") {
    group = "verification"
    description = "Verify SECURITY.md and CONTRIBUTING.md exist"
    doLast {
        listOf("SECURITY.md", "CONTRIBUTING.md").forEach { name ->
            val f = file(name)
            require(f.exists()) { "$name not found in ${projectDir.absolutePath}" }
            require(f.length() > 0) { "$name is empty" }
            println("  $name exists (${f.length()} bytes)")
        }
    }
}

tasks.register("resolveAndLockAll") {
    group = "verification"
    description = "Resolve all dependencies and write lockfiles (run with --write-locks)"
    dependsOn(libraryProjects.map { "${it.path}:resolveAndLockAllConfigurations" })
}

tasks.register("verifyLockfiles") {
    group = "verification"
    description = "Verify dependency lockfiles exist for all library modules"
    doLast {
        libraryProjects.forEach { proj ->
            val lockfile = proj.file("gradle.lockfile")
            require(lockfile.exists()) { "Missing lockfile: ${lockfile.absolutePath}" }
            println("  Lockfile: ${proj.name}")
        }
    }
}

tasks.register("verifyPublishingConfigured") {
    group = "verification"
    description = "Verify Maven publications and JReleaser configuration for release modules"
    doLast {
        require(file("jreleaser.yml").isFile) { "jreleaser.yml not found" }
        publishedProjects.forEach { proj ->
            require(proj.plugins.hasPlugin("maven-publish")) {
                "maven-publish plugin not applied to ${proj.name}"
            }
            val publishing = proj.extensions.getByType<PublishingExtension>()
            require("maven" in publishing.publications.names) {
                "Maven publication not configured for ${proj.name}"
            }
            println("  Maven publication configured: ${proj.name}")
        }
    }
}

tasks.register("verifyFacadeRuntimeFootprint") {
    group = "verification"
    description = "Verify the vectors facade stays lightweight and free of optional integrations"
    dependsOn(":vectors:verifyRuntimeFootprint")
}

tasks.register("verifyStagedPublications") {
    group = "verification"
    description = "Stage and validate every Maven Central artifact and its internal dependencies"
    dependsOn(publishedProjects.map { "${it.path}:publishMavenPublicationToStagingRepository" })
    doLast {
        val releaseVersion = project.version.toString()
        val stagingRoot = layout.buildDirectory.dir("staging-deploy/com/integrallis").get().asFile
        val internalDependency = Regex(
            """<dependency>\s*<groupId>com\.integrallis</groupId>\s*<artifactId>([^<]+)</artifactId>"""
        )
        publishedProjects.forEach { proj ->
            val versionDir = stagingRoot.resolve("${proj.name}/$releaseVersion")
            val pomFile = versionDir.listFiles()
                ?.filter { it.isFile && it.name.endsWith(".pom") }
                ?.maxByOrNull { it.lastModified() }
                ?: error("Missing staged POM in $versionDir")
            val artifactBase = pomFile.name.removeSuffix(".pom")
            listOf(
                "$artifactBase.jar",
                "$artifactBase-sources.jar",
                "$artifactBase-javadoc.jar"
            ).forEach { name ->
                require(versionDir.resolve(name).isFile) {
                    "Missing staged artifact: ${versionDir.resolve(name)}"
                }
            }

            val pom = pomFile.readText()
            require("<licenses>" in pom && "<developers>" in pom && "<scm>" in pom) {
                "Incomplete Maven Central metadata in ${proj.name} POM"
            }
            when (proj.name) {
                "vectors-storage" ->
                    require("<groupId>software.amazon.awssdk</groupId>" !in pom) {
                        "vectors-storage must not publish the opt-in AWS SDK dependency"
                    }
                "vectors-db" ->
                    require("<groupId>org.apache.arrow</groupId>" !in pom) {
                        "vectors-db must not publish the opt-in Apache Arrow dependencies"
                    }
                "vectors-storage-s3" ->
                    require(
                        "<artifactId>vectors-storage</artifactId>" in pom &&
                            "<artifactId>s3</artifactId>" in pom
                    ) {
                        "vectors-storage-s3 must publish both storage and AWS S3 dependencies"
                    }
                "vectors-db-arrow" ->
                    require(
                        "<artifactId>vectors-db</artifactId>" in pom &&
                            "<artifactId>arrow-vector</artifactId>" in pom &&
                            "<artifactId>arrow-memory-unsafe</artifactId>" in pom
                    ) {
                        "vectors-db-arrow must publish the database and Arrow runtime dependencies"
                    }
            }
            internalDependency.findAll(pom).forEach { match ->
                val artifactId = match.groupValues[1]
                require(artifactId in publishedModuleNames) {
                    "${proj.name} publishes an unavailable internal dependency: $artifactId"
                }
            }
            println("  Staged publication valid: ${proj.name}")
        }
    }
}

tasks.register("verifyReproducibleBuild") {
    group = "verification"
    description = "Verify JAR tasks are configured for reproducible builds"
    // No dependsOn(:jar) — we only inspect task configuration, not outputs.
    // This avoids a Gradle 9 lifecycle conflict with CycloneDX task creation.
    doLast {
        libraryProjects.forEach { proj ->
            proj.tasks.withType<Jar>().forEach { jar ->
                require(!jar.isPreserveFileTimestamps) {
                    "preserveFileTimestamps must be false for ${proj.name}:${jar.name}"
                }
                require(jar.isReproducibleFileOrder) {
                    "reproducibleFileOrder must be true for ${proj.name}:${jar.name}"
                }
            }
            println("  Reproducible JARs: ${proj.name}")
        }
    }
}

tasks.register("verifyGithubWorkflows") {
    group = "verification"
    description = "Verify GitHub Actions workflow files exist"
    doLast {
        val workflowDir = rootProject.file(".github/workflows")
        listOf("ci.yml", "docs.yml", "scorecard.yml", "mfcqi.yml", "release.yml")
            .forEach { name ->
            val f = workflowDir.resolve(name)
            require(f.exists()) { "Missing workflow: ${f.absolutePath}" }
            val content = f.readText()
            require(content.contains("jobs:")) { "$name missing 'jobs:' section" }
            println("  Workflow: $name")
        }
        val releaseWorkflow = workflowDir.resolve("release.yml").readText()
        require("libopenblas-dev" in releaseWorkflow) {
            "release.yml runs the all-project build and must install OpenBLAS for Vectors Studio"
        }
    }
}

tasks.register("verifyNotebooks") {
    group = "verification"
    description = "Verify that JJava notebooks are portable, current, and free of checked-in errors"

    val notebookFiles = fileTree("notebooks") {
        include("*.ipynb")
    }
    val kernelFile = file("notebooks/jupyter/kernel.json")
    val notebookReadme = file("notebooks/README.md")
    inputs.files(notebookFiles, kernelFile, notebookReadme)

    doLast {
        val forbiddenText = mapOf(
            "0.1.0-SNAPSHOT" to "hardcoded snapshot dependency",
            "/build/libs/" to "hardcoded module JAR path",
            "%classpath" to "notebook-local classpath setup",
            "%%loadFromPOM" to "notebook-local Maven dependency setup",
            "com.integrallis.vectors.db.Document" to "stale Document package",
            "langchain4j-core:1.0.0-beta1" to "outdated LangChain4j dependency"
        )
        val expectedOutput = mapOf(
            "01_getting_started.ipynb" to listOf("collection size: 500"),
            "02_quantization_tour.ipynb" to listOf("recall regression floors satisfied"),
            "03_spring_ai_integration.ipynb" to
                listOf("SIMD via Panama Vector API", "after delete, collection size: 4"),
            "04_langchain4j_integration.ipynb" to listOf("[1] score=", "SIMD via Panama"),
            "05_embedding_cache.ipynb" to
                listOf("delegateCalls=4", "delegateCalls=5"),
            "06_vcr_test_harness.ipynb" to
                listOf("model=text-embedding-3-small", "vec[0]=0.110000")
        )

        notebookFiles.files.sorted().forEach { notebookFile ->
            val notebook = JsonSlurper().parse(notebookFile) as Map<*, *>
            val metadata = notebook["metadata"] as? Map<*, *> ?: emptyMap<Any, Any>()
            val kernelSpec = metadata["kernelspec"] as? Map<*, *> ?: emptyMap<Any, Any>()
            require(kernelSpec["name"] == "java") {
                "${notebookFile.name} must use the JJava kernel"
            }

            val cells = notebook["cells"] as? List<*> ?: emptyList<Any>()
            fun fragments(value: Any?): String = when (value) {
                is List<*> -> value.joinToString("") { it.toString() }
                null -> ""
                else -> value.toString()
            }
            val executionSourceDigest = MessageDigest.getInstance("SHA-256")
            cells.mapNotNull { it as? Map<*, *> }.forEach { cell ->
                executionSourceDigest.update(cell["cell_type"].toString().toByteArray())
                executionSourceDigest.update(0)
                executionSourceDigest.update(fragments(cell["source"]).toByteArray())
                executionSourceDigest.update(0)
            }
            val executionSourceSha256 =
                executionSourceDigest.digest().joinToString("") { "%02x".format(it) }
            val vectorsMetadata =
                metadata["vectors"] as? Map<*, *> ?: emptyMap<Any, Any>()
            require(vectorsMetadata["execution_source_sha256"] == executionSourceSha256) {
                "${notebookFile.name} outputs were not executed from its current cell sources"
            }

            val codeCells = cells
                .mapNotNull { it as? Map<*, *> }
                .filter { it["cell_type"] == "code" }
            require(codeCells.isNotEmpty()) {
                "${notebookFile.name} must contain executable code"
            }
            require(codeCells.all { it["execution_count"] is Number }) {
                "${notebookFile.name} must ship with every code cell executed"
            }

            val source = cells.joinToString("\n") { cell ->
                val sourceLines = (cell as? Map<*, *>)?.get("source") as? List<*> ?: emptyList<Any>()
                sourceLines.joinToString("") { it.toString() }
            }
            forbiddenText.forEach { (needle, description) ->
                require(needle !in source) {
                    "${notebookFile.name} contains $description: $needle"
                }
            }

            val outputs = codeCells.flatMap { cell ->
                (cell["outputs"] as? List<*> ?: emptyList<Any>())
                    .mapNotNull { it as? Map<*, *> }
            }
            val checkedInErrors = outputs.filter { it["output_type"] == "error" }
            require(checkedInErrors.isEmpty()) {
                "${notebookFile.name} contains ${checkedInErrors.size} checked-in execution error(s)"
            }
            val checkedInStderr = outputs.filter {
                it["output_type"] == "stream" && it["name"] == "stderr"
            }
            require(checkedInStderr.isEmpty()) {
                "${notebookFile.name} contains ${checkedInStderr.size} checked-in stderr output(s)"
            }

            val renderedOutput = outputs.joinToString("") { output ->
                when (output["output_type"]) {
                    "stream" -> fragments(output["text"])
                    "execute_result", "display_data" ->
                        fragments((output["data"] as? Map<*, *>)?.get("text/plain"))
                    else -> ""
                }
            }
            expectedOutput.getValue(notebookFile.name).forEach { expected ->
                require(expected in renderedOutput) {
                    "${notebookFile.name} is missing checked-in output: $expected"
                }
            }
            println("  Notebook: ${notebookFile.name}")
        }

        val kernel = JsonSlurper().parse(kernelFile) as Map<*, *>
        val kernelEnv = kernel["env"] as? Map<*, *> ?: emptyMap<Any, Any>()
        require(kernelEnv["JJAVA_CLASSPATH"] ==
            "/home/jovyan/work/vectors/build/notebooks/classpath/*") {
            "JJava kernel must load the generated notebook classpath"
        }

        val readmeText = notebookReadme.readText()
        require("install.sh" !in readmeText) {
            "Notebook README references the nonexistent jupyter/install.sh"
        }
    }
}

tasks.register("verifyDocumentation") {
    group = "verification"
    description = "Verify README, Antora content, landing page, and documentation toolchain"

    val readmeFile = file("README.md")
    val docsFiles = fileTree("docs") {
        include("**/*.adoc", "**/*.html", "**/*.hbs", "**/*.yml", "**/*.yaml")
        exclude("build/**", "node_modules/**", ".gradle/**", "content/**/attachments/**")
    }
    val docsPackageFile = file("docs/package.json")
    val docsLockFile = file("docs/package-lock.json")
    val docsBuildFile = file("docs/build.gradle")
    val ciWorkflowFile = file(".github/workflows/ci.yml")
    val docsWorkflowFile = file(".github/workflows/docs.yml")
    val releaseWorkflowFile = file(".github/workflows/release.yml")
    inputs.files(
        readmeFile,
        docsFiles,
        docsPackageFile,
        docsLockFile,
        docsBuildFile,
        ciWorkflowFile,
        docsWorkflowFile,
        releaseWorkflowFile
    )

    doLast {
        val staleRepositoryUrl = "https://github.com/integrallis/java-vectors"
        val staleReferences = docsFiles.files.filter { staleRepositoryUrl in it.readText() }
        require(staleReferences.isEmpty()) {
            "Documentation contains stale repository URLs: " +
                staleReferences.joinToString { it.relativeTo(rootDir).path }
        }

        val readme = readmeFile.readText()
        publishedModuleNames.forEach { module ->
            require("`$module`" in readme) {
                "README module inventory is missing $module"
            }
        }
        require("https://integrallis.github.io/vectors/" in readme) {
            "README must link to the published documentation site"
        }
        require("notebooks/README.md" in readme) {
            "README must link to the executable notebooks"
        }

        val docsIndex = file("docs/content/modules/ROOT/pages/index.adoc").readText()
        publishedModuleNames.forEach { module ->
            require("`$module`" in docsIndex) {
                "Documentation module inventory is missing $module"
            }
        }

        val docsNav = file("docs/content/modules/ROOT/nav.adoc").readText()
        require("notebooks.adoc" in docsNav) {
            "Antora navigation must link to the notebook guide"
        }
        require("studio.adoc" in docsNav) {
            "Antora navigation must link to the Vectors Studio guide"
        }

        val landingPage = file("docs/landing/index.html").readText()
        require("collection.search(query, 10)" !in landingPage) {
            "Landing page uses a nonexistent VectorCollection.search overload"
        }

        require(docsLockFile.exists()) {
            "docs/package-lock.json must be committed for reproducible documentation builds"
        }
        val docsPackage = JsonSlurper().parse(docsPackageFile) as Map<*, *>
        val dependencies = docsPackage["dependencies"] as? Map<*, *> ?: emptyMap<Any, Any>()
        listOf("@antora/cli", "@antora/site-generator", "@antora/site-generator-default")
            .forEach { dependency ->
                require(dependencies[dependency] == "3.1.12") {
                    "$dependency must be pinned exactly to 3.1.12"
                }
            }
        val overrides = docsPackage["overrides"] as? Map<*, *> ?: emptyMap<Any, Any>()
        require(overrides["js-yaml"] == "4.3.0") {
            "Documentation dependencies must override js-yaml to 4.3.0"
        }
        require("args = ['ci']" in docsBuildFile.readText()) {
            "Documentation build must use npm ci"
        }
        require(":docs:build" in ciWorkflowFile.readText()) {
            "CI must execute the documentation build"
        }
        require("test-notebooks.sh" in ciWorkflowFile.readText()) {
            "CI must execute and validate the Java notebooks"
        }
        require(":docs:build" in docsWorkflowFile.readText()) {
            "GitHub Pages workflow must execute the documentation build"
        }
        require(":docs:build" in releaseWorkflowFile.readText()) {
            "Release workflow must execute the documentation build"
        }
        require("-PnotebookMode=release" in releaseWorkflowFile.readText()) {
            "Release workflow must validate notebooks against staged release artifacts"
        }
    }
}

tasks.register<Sync>("prepareNotebookClasspath") {
    group = "documentation"
    description = "Prepare the JJava runtime classpath from source or released artifacts"

    val selectedMode = notebookMode.map { it.lowercase() }
    val selectedClasspath = when (selectedMode.get()) {
        "source" -> notebookSourceClasspath
        "release" -> notebookReleaseClasspath
        else -> throw GradleException(
            "Unsupported notebookMode '${notebookMode.get()}'; expected 'source' or 'release'"
        )
    }

    from(selectedClasspath)
    into(layout.buildDirectory.dir("notebooks/classpath"))
    duplicatesStrategy = DuplicatesStrategy.FAIL
    inputs.property("notebookMode", selectedMode)
    inputs.property("notebookVersion", notebookVersion)
    inputs.property("notebookRepository", notebookRepositoryUrl.orElse(""))

    doLast {
        val details = if (selectedMode.get() == "release") {
            "com.integrallis ${notebookVersion.get()}"
        } else {
            "current project sources"
        }
        println("  Notebook classpath: ${destinationDir.absolutePath}")
        println("  Dependency mode: $details")
    }
}

tasks.register("complianceCheck") {
    group = "verification"
    description = "Run all compliance verification tasks"
    dependsOn(
        "verifySbom",
        "verifyGovernanceFiles",
        "verifyLockfiles",
        "verifyPublishingConfigured",
        "verifyFacadeRuntimeFootprint",
        "verifyStagedPublications",
        "verifyReproducibleBuild",
        "verifyGithubWorkflows",
        "verifyNotebooks",
        "verifyDocumentation"
    )
}

tasks.wrapper {
    gradleVersion = "9.4.1"
    distributionType = Wrapper.DistributionType.ALL
}

// Resolvable classpath for aggregateJavadoc. Declaring it as a root-project configuration that
// depends on each library module routes classpath resolution through Gradle's normal dependency
// machinery — which is configuration-cache safe. The prior approach (reading each subproject's
// sourceSets.main.compileClasspath inside the task's configure block) resolved other projects'
// configurations at configuration time without an exclusive lock, which the configuration cache
// rejects with a serialization error.
val aggregateJavadocClasspath: Configuration by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
}

dependencies {
    publishedProjects.forEach { aggregateJavadocClasspath(project(it.path)) }
    // vectors-db keeps its optional cuVS adapter compile-only so CPU consumers never resolve GPU
    // artifacts. Javadoc still parses those adapter sources and therefore needs the GPU types on
    // its documentation-only classpath.
    aggregateJavadocClasspath(project(":vectors-gpu"))
    aggregateJavadocClasspath("dev.langchain4j:langchain4j-core:1.13.1")
    aggregateJavadocClasspath("org.springframework.ai:spring-ai-vector-store:1.1.4")
    aggregateJavadocClasspath("org.springframework.ai:spring-ai-model:1.1.4")
    aggregateJavadocClasspath("org.springframework.boot:spring-boot-autoconfigure:3.4.5")
    aggregateJavadocClasspath("io.projectreactor:reactor-core:3.7.9")
    aggregateJavadocClasspath("org.junit.jupiter:junit-jupiter-api:5.11.4")
    aggregateJavadocClasspath("org.apiguardian:apiguardian-api:1.1.2")
    aggregateJavadocClasspath("org.testng:testng:7.10.2")
}

val cleanCollectedJavadocs by tasks.registering(Delete::class) {
    description = "Remove stale Javadocs before collecting the published API"
    group = "documentation"
    delete(layout.buildDirectory.dir("docs/javadoc"))
}

// Aggregated Javadoc generation
tasks.register<Javadoc>("aggregateJavadoc") {
    description = "Generate aggregated Javadoc for all library modules"
    group = "documentation"
    dependsOn(cleanCollectedJavadocs)
    javadocTool.set(
        javaToolchains.javadocToolFor {
            languageVersion = JavaLanguageVersion.of(25)
        }
    )

    val libProjects = publishedProjects
    // Library projects are evaluated before :docs (the caller), so the main source sets are
    // already realised by the time this task's configure lambda runs. Using afterEvaluate here
    // fails under Gradle 9 because the callback would fire on an already-evaluated project.
    libProjects.forEach { proj ->
        dependsOn(proj.tasks.named("compileJava"))
        source(proj.the<SourceSetContainer>()["main"].allJava)
    }
    classpath = aggregateJavadocClasspath
    setDestinationDir(layout.buildDirectory.dir("docs/javadoc/aggregate").get().asFile)

    (options as StandardJavadocDocletOptions).apply {
        title = "Vectors ${project.version} API"
        windowTitle = "Vectors ${project.version}"
        author(true)
        version(true)
        use(true)
        splitIndex(true)
        links("https://docs.oracle.com/en/java/javase/25/docs/api/")
        addStringOption("Xdoclint:-missing", "-quiet")
        addStringOption("-add-modules", "jdk.incubator.vector")
    }

    isFailOnError = true
}

// Per-module Javadocs
// Per-module Javadoc collection. Implemented as one standard Sync task per module rather than a
// single doLast that loops over Project references and calls project.copy(...): a task action
// may not capture Project objects or use the project copy API under the configuration cache.
// Sync is a built-in, configuration-cache-compatible task type; wiring `from` to each module's
// javadoc task carries the cross-project dependency without serializing a Project.
val perModuleJavadocCopies =
    publishedProjects
        .map { proj ->
            val moduleName = proj.name.removePrefix("vectors-")
            tasks.register<Sync>("copyJavadoc_$moduleName") {
                description = "Collects $moduleName Javadoc into the aggregate docs tree"
                group = "documentation"
                dependsOn(cleanCollectedJavadocs)
                from(proj.tasks.named<Javadoc>("javadoc"))
                into(layout.buildDirectory.dir("docs/javadoc/modules/$moduleName"))
            }
        }

tasks.register("generateModuleJavadocs") {
    description = "Generate Javadoc for individual library modules"
    group = "documentation"
    dependsOn(perModuleJavadocCopies)
}
