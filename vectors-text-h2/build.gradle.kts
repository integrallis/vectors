description = "H2 embedded database implementation of TextIndexSpi"

dependencies {
    api(project(":vectors-hybrid"))
    implementation("com.h2database:h2:2.3.232")
}
