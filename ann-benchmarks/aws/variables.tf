# Copyright 2025-2026 Integrallis Software, LLC. Apache-2.0.

variable "region" {
  description = "AWS region to launch the benchmark instance in."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type. Pick ONE and keep it constant for reproducible, comparable
    numbers. A compute/memory-optimized box with enough RAM to hold the dataset +
    indexes is right; the ANN-Benchmarks suite is single-threaded by design, so
    huge vCPU counts are wasted. Examples: c7i.4xlarge, r7i.4xlarge.
  EOT
  type        = string
  default     = "c7i.4xlarge"
}

variable "disk_gb" {
  description = "Root EBS volume size (GB). Datasets + Docker images are tens of GB."
  type        = number
  default     = 200
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH access."
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in (lock this down to your IP)."
  type        = string
  default     = "0.0.0.0/0"
}

variable "s3_results_uri" {
  description = "Optional s3://bucket/prefix to upload results to. Empty disables upload."
  type        = string
  default     = ""
}

variable "s3_results_bucket" {
  description = "Bucket name granted to the instance role for results upload (the bucket part of s3_results_uri). Empty disables the S3 IAM policy."
  type        = string
  default     = ""
}

variable "vectors_repo" {
  description = "Git URL of the vectors repo to build."
  type        = string
  default     = "https://github.com/integrallis/vectors.git"
}

variable "vectors_ref" {
  description = "vectors branch/tag/sha to build."
  type        = string
  default     = "main"
}

variable "dataset" {
  description = "ANN-Benchmarks dataset (e.g. sift-128-euclidean, glove-100-angular, gist-960-euclidean)."
  type        = string
  default     = "sift-128-euclidean"
}

variable "algorithms" {
  description = "Space-separated algorithms to run. Add competitors for a head-to-head, e.g. \"vectors hnswlib faiss\"."
  type        = string
  default     = "vectors"
}
