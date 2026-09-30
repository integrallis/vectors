description = "Jackson CassetteSerializer"

dependencies {
    api(project(":vectors-vcr-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.7")
    testImplementation("com.fasterxml.jackson.core:jackson-databind:2.21.7")
    testImplementation(project(":vectors-vcr-serde-avaje"))
}
