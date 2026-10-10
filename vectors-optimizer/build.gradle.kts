description = "Hyperparameter optimization for vectors-* — index/router/cache studies with Grid/Random/TPE samplers"

val jacksonVersion = rootProject.extra["jacksonVersion"] as String

dependencies {
    api(project(":vectors-core"))
    implementation(project(":vectors-db"))
    implementation(project(":vectors-cache"))
    implementation(project(":vectors-cache-semantic-db"))
    implementation(project(":vectors-router"))
    implementation(project(":vectors-bench"))

    implementation(platform("com.fasterxml.jackson:jackson-bom:$jacksonVersion"))
    implementation("com.fasterxml.jackson.core:jackson-databind")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310")
}
