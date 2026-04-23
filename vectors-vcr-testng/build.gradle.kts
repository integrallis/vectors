description = "TestNG listener for VCR record/replay"

dependencies {
    api(project(":vectors-vcr-core"))
    compileOnly("org.testng:testng:7.10.2")
    testImplementation("org.testng:testng:7.10.2")
    testImplementation(project(":vectors-vcr-serde-avaje"))
}

// Module's own tests use TestNG (JUnit 5 remains the default across the rest of the repo).
tasks.named<Test>("test") {
    useTestNG {
        excludeGroups("slow", "benchmark", "integration")
    }
    filter {
        excludeTestsMatching("*Scenario")
    }
}

tasks.named<Test>("unitTest") {
    useTestNG {
        includeGroups("unit")
    }
}

tasks.named<Test>("slowTest") {
    useTestNG {
        includeGroups("slow")
    }
}

tasks.named<Test>("integrationTest") {
    useTestNG {
        includeGroups("integration")
    }
}
