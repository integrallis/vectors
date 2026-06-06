# Copyright 2025-2026 Integrallis Software, LLC. Apache-2.0.
#
# Provisions ONE EC2 instance that runs the ANN-Benchmarks harness for `vectors`
# (via run-benchmark.sh in this directory) and optionally ships results to S3.
# Single-node by design — no Kubernetes, no autoscaling. Destroy with
# `terraform destroy` when the run finishes.

terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Latest Canonical Ubuntu 22.04 LTS AMI (matches the ann-benchmarks base OS).
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "bench" {
  name_prefix = "vectors-bench-"
  description = "vectors ANN-Benchmarks runner: SSH in, all out"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Instance role — only created with an S3 policy when a results bucket is given.
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "bench" {
  name_prefix        = "vectors-bench-"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy" "s3" {
  count = var.s3_results_bucket == "" ? 0 : 1
  name  = "results-upload"
  role  = aws_iam_role.bench.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.s3_results_bucket}",
        "arn:aws:s3:::${var.s3_results_bucket}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "bench" {
  name_prefix = "vectors-bench-"
  role        = aws_iam_role.bench.name
}

resource "aws_instance" "bench" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.bench.id]
  iam_instance_profile   = aws_iam_instance_profile.bench.name

  root_block_device {
    volume_size = var.disk_gb
    volume_type = "gp3"
  }

  # Bootstrap: clone the repo for run-benchmark.sh, then run it as root. Progress
  # streams to /var/log/vectors-bench.log; results land under /opt/vectors-bench.
  user_data = <<-EOT
    #!/usr/bin/env bash
    set -euxo pipefail
    apt-get update -y && apt-get install -y git
    git clone --branch '${var.vectors_ref}' '${var.vectors_repo}' /opt/vectors-src \
      || git clone '${var.vectors_repo}' /opt/vectors-src
    export VECTORS_REPO=/opt/vectors-src
    export VECTORS_REF='${var.vectors_ref}'
    export DATASET='${var.dataset}'
    export ALGORITHMS='${var.algorithms}'
    export S3_RESULTS='${var.s3_results_uri}'
    export WORKDIR=/opt/vectors-bench
    bash /opt/vectors-src/ann-benchmarks/aws/run-benchmark.sh \
      >> /var/log/vectors-bench.log 2>&1
  EOT

  tags = {
    Name = "vectors-ann-benchmarks"
  }
}
