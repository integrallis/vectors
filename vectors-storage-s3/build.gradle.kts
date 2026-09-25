description = "Opt-in AWS SDK runtime for S3-compatible Vectors storage"

dependencies {
    api(platform("io.netty:netty-bom:4.1.138.Final"))
    api(project(":vectors-storage"))
    api("software.amazon.awssdk:s3:2.29.52")
}
