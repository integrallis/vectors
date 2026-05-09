description = "Optimizer tutorial: end-to-end before/after on a real ANN-Benchmarks dataset"

dependencies {
    implementation(project(":vectors-core"))
    implementation(project(":vectors-db"))
    implementation(project(":vectors-router"))
    implementation(project(":vectors-optimizer"))
    implementation(project(":vectors-bench"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.18.2")
}

application {
    mainClass.set("com.integrallis.vectors.demo.optimizer.TutorialApp")
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}

tasks.register<JavaExec>("stage1Baseline") {
    group = "tutorial"
    description = "Stage 1 — Build with default HNSW params, record baseline metrics."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.demo.optimizer.Stage1Baseline")
}

tasks.register<JavaExec>("stage2BroadSweep") {
    group = "tutorial"
    description = "Stage 2 — 20-trial Random sweep over m / efConstruction."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.demo.optimizer.Stage2BroadSweep")
}

tasks.register<JavaExec>("stage3Tpe") {
    group = "tutorial"
    description = "Stage 3 — 40-trial TPE study; prints side-by-side comparison vs. baseline."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.demo.optimizer.Stage3TpeOptimization")
}

tasks.register<JavaExec>("stage4Threshold") {
    group = "tutorial"
    description = "Stage 4 — RouterThresholdStudy on a small labelled probe set."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.integrallis.vectors.demo.optimizer.Stage4RouterThreshold")
}
