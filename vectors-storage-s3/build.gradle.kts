description = "Opt-in AWS SDK runtime for S3-compatible Vectors storage"

dependencies {
    api(project(":vectors-storage"))
    api("software.amazon.awssdk:s3:2.29.52")
}
