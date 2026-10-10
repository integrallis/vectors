description = "HTTP client for the vectors-server REST API"

val jacksonVersion = rootProject.extra["jacksonVersion"] as String

dependencies {
    api(project(":vectors-core"))
    api("com.fasterxml.jackson.core:jackson-databind:$jacksonVersion")
    api("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:$jacksonVersion")
}
