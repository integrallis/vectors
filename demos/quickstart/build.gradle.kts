description = "Quickstart demo: open a VectorCollection, add vectors, search"

dependencies {
    implementation(project(":vectors-db"))
}

application {
    mainClass.set("com.integrallis.vectors.demo.quickstart.QuickstartApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}
