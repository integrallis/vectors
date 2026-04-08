plugins {
    java
    `java-library`
    `maven-publish`
    id("com.github.spotbugs") version "6.4.4" apply false
    id("com.diffplug.spotless") version "6.25.0" apply false
    jacoco
}

allprojects {
    group = "com.integrallis"

    repositories {
        mavenCentral()
    }
}

// Library subprojects (excludes docs)
val libraryProjects = subprojects.filter { it.name != "docs" }

configure(libraryProjects) {
    apply(plugin = "java")
    apply(plugin = "java-library")
    apply(plugin = "maven-publish")
    apply(plugin = "com.github.spotbugs")
    apply(plugin = "com.diffplug.spotless")
    apply(plugin = "jacoco")

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
            "-Werror",
            "--add-modules", "jdk.incubator.vector"
        ))
    }

    tasks.withType<Test> {
        useJUnitPlatform {
            excludeTags("slow", "benchmark")
        }
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
        jvmArgs("--add-modules", "jdk.incubator.vector")
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
        jvmArgs("--add-modules", "jdk.incubator.vector")
        testLogging {
            events("passed", "skipped", "failed")
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

tasks.wrapper {
    gradleVersion = "9.4.1"
    distributionType = Wrapper.DistributionType.ALL
}

// Aggregated Javadoc generation
tasks.register<Javadoc>("aggregateJavadoc") {
    description = "Generate aggregated Javadoc for all library modules"
    group = "documentation"

    val libProjects = libraryProjects.filter { it.name != "vectors-bench" }
    libProjects.forEach { proj ->
        proj.afterEvaluate {
            source(proj.the<SourceSetContainer>()["main"].allJava)
            classpath += files(proj.the<SourceSetContainer>()["main"].compileClasspath)
        }
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
