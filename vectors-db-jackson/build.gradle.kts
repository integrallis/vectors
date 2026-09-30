description = "Jackson-backed RecipeCodec for vectors: opt-in, so the core library carries no JSON dependency"

dependencies {
    api(project(":vectors-core"))
    // 2.21.7 matches the other published modules that ship Jackson (vectors-db-arrow,
    // vectors-vcr-serde-jackson), so a Maven consumer importing any combination resolves one version.
    implementation(platform("com.fasterxml.jackson:jackson-bom:2.21.7"))
    implementation("com.fasterxml.jackson.core:jackson-databind")
}
