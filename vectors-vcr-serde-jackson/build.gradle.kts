description = "Jackson CassetteSerializer"

dependencies {
    api(project(":vectors-vcr-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:${rootProject.extra["jacksonVersion"]}")
    testImplementation("com.fasterxml.jackson.core:jackson-databind:${rootProject.extra["jacksonVersion"]}")
    testImplementation(project(":vectors-vcr-serde-avaje"))
}
