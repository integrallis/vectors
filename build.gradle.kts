plugins {
    java
    `java-library`
    `maven-publish`
    id("com.github.spotbugs") version "6.4.4" apply false
    id("com.diffplug.spotless") version "6.25.0" apply false
    id("org.cyclonedx.bom") version "3.2.4" apply false
    id("org.owasp.dependencycheck") version "12.2.1" apply false
    id("dev.sigstore.sign") version "2.0.0" apply false
    jacoco
}

allprojects {
    group = "com.integrallis"

    repositories {
        mavenCentral()
    }
}

// Library subprojects (excludes docs and runnable demos)
val libraryProjects = subprojects.filter {
    it.name != "docs" && !it.path.startsWith(":demos:")
}
val demoProjects = subprojects.filter { it.path.startsWith(":demos:") }

// FSL-1.1-ALv2 modules — all others are Apache 2.0
val fslModules = setOf("vectors-distributed", "vectors-server", "vectors-gpu")

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
    apply(plugin = "maven-publish")
    apply(plugin = "com.github.spotbugs")
    apply(plugin = "com.diffplug.spotless")
    apply(plugin = "jacoco")
    apply(plugin = "org.cyclonedx.bom")
    apply(plugin = "org.owasp.dependencycheck")
    apply(plugin = "dev.sigstore.sign")

    // Dependency locking — enforced when lockfiles exist, lenient otherwise
    dependencyLocking {
        lockAllConfigurations()
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
        maxParallelForks = Runtime.getRuntime().availableProcessors()
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
        maxParallelForks = 1
        systemProperty("junit.jupiter.execution.timeout.default", "30m")
    }

    // Default 'test' task excludes all infrastructure-heavy tags.
    tasks.named<Test>("test") {
        useJUnitPlatform {
            excludeTags("slow", "benchmark", "integration", "distributed", "k8s", "chaos", "scale")
        }
    }

    tasks.withType<Javadoc> {
        val javadocOptions = options as StandardJavadocDocletOptions
        javadocOptions.addBooleanOption("Xdoclint:all,-missing", true)
        javadocOptions.addBooleanOption("html5", true)
        javadocOptions.addStringOption("-add-modules", "jdk.incubator.vector")
        isFailOnError = false
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
        violationRules {
            rule {
                limit {
                    minimum = "0.80".toBigDecimal()
                }
            }
        }
    }

    // OWASP Dependency-Check — runs only when explicitly invoked
    configure<org.owasp.dependencycheck.gradle.extension.DependencyCheckExtension> {
        failBuildOnCVSS = 7.0f
        formats.set(listOf("HTML", "JSON", "SARIF"))
        outputDirectory.set(layout.buildDirectory.dir("reports/dependency-check"))
        nvd.apiKey = System.getenv("NVD_API_KEY") ?: ""
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
        implementation("org.slf4j:slf4j-api:2.0.16")

        // Testing
        testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
        testImplementation("org.assertj:assertj-core:3.27.2")
        testImplementation("org.mockito:mockito-core:5.15.2")
        testImplementation("org.mockito:mockito-junit-jupiter:5.15.2")
        testRuntimeOnly("org.junit.platform:junit-platform-launcher")
        testRuntimeOnly("ch.qos.logback:logback-classic:1.5.15")
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
    description = "Verify CycloneDX SBOM generation for all library modules"
    dependsOn(libraryProjects.map { "${it.path}:cyclonedxDirectBom" })
    doLast {
        libraryProjects.forEach { proj ->
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
    notCompatibleWithConfigurationCache("Resolves and locks configurations")
    doFirst {
        require(gradle.startParameter.isWriteDependencyLocks) {
            "${path} must be run with the --write-locks flag"
        }
    }
    doLast {
        libraryProjects.forEach { proj ->
            proj.configurations.filter { it.isCanBeResolved }.forEach { config ->
                try {
                    config.resolve()
                } catch (_: Exception) {
                    // Some configurations may not be resolvable — skip them
                }
            }
        }
    }
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

tasks.register("verifySigningConfigured") {
    group = "verification"
    description = "Verify Sigstore signing plugin is applied to all library modules"
    doLast {
        libraryProjects.forEach { proj ->
            require(proj.plugins.hasPlugin("dev.sigstore.sign")) {
                "Sigstore plugin not applied to ${proj.name}"
            }
            println("  Sigstore configured: ${proj.name}")
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
        listOf("ci.yml", "scorecard.yml", "codeql.yml").forEach { name ->
            val f = workflowDir.resolve(name)
            require(f.exists()) { "Missing workflow: ${f.absolutePath}" }
            val content = f.readText()
            require(content.contains("jobs:")) { "$name missing 'jobs:' section" }
            println("  Workflow: $name")
        }
    }
}

tasks.register("complianceCheck") {
    group = "verification"
    description = "Run all compliance verification tasks"
    dependsOn(
        "verifySbom",
        "verifyGovernanceFiles",
        "verifySigningConfigured",
        "verifyReproducibleBuild"
    )
}

tasks.wrapper {
    gradleVersion = "9.4.1"
    distributionType = Wrapper.DistributionType.ALL
}

// Aggregated Javadoc generation
tasks.register<Javadoc>("aggregateJavadoc") {
    description = "Generate aggregated Javadoc for all library modules"
    group = "documentation"

    val libProjects = libraryProjects.filter { it.name != "vectors-bench" }
    // Library projects are evaluated before :docs (the caller), so the main source sets are
    // already realised by the time this task's configure lambda runs. Using afterEvaluate here
    // fails under Gradle 9 because the callback would fire on an already-evaluated project.
    libProjects.forEach { proj ->
        dependsOn(proj.tasks.named("compileJava"))
        source(proj.the<SourceSetContainer>()["main"].allJava)
        classpath += files(proj.the<SourceSetContainer>()["main"].compileClasspath)
    }
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

    isFailOnError = false
}

// Per-module Javadocs
tasks.register("generateModuleJavadocs") {
    description = "Generate Javadoc for individual library modules"
    group = "documentation"

    val libProjects = libraryProjects.filter { it.name != "vectors-bench" }
    libProjects.forEach { proj ->
        dependsOn(proj.tasks.named("javadoc"))
    }

    doLast {
        libProjects.forEach { proj ->
            val moduleName = proj.name.removePrefix("vectors-")
            copy {
                from(proj.tasks.named<Javadoc>("javadoc").get().destinationDir)
                into(layout.buildDirectory.dir("docs/javadoc/modules/$moduleName").get().asFile)
            }
        }
    }
}
