description = "Jackson CassetteSerializer"

dependencies {
    api(project(":vectors-vcr-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.18.2")
    testImplementation("com.fasterxml.jackson.core:jackson-databind:2.18.2")
    testImplementation(project(":vectors-vcr-serde-avaje"))
}
