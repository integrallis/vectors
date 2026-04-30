description = "HTTP client for the vectors-server REST API"

val jacksonVersion = "2.18.2"

dependencies {
    api(project(":vectors-core"))
    api("com.fasterxml.jackson.core:jackson-databind:$jacksonVersion")
    api("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:$jacksonVersion")
}
