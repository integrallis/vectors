description = "JSR-107 JCache bridge for the VectorCache SPI"

dependencies {
    api(project(":vectors-cache"))
    api("javax.cache:cache-api:1.1.1")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testImplementation("com.github.ben-manes.caffeine:jcache:3.1.8")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
