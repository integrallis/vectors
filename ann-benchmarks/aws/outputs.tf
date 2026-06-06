# Copyright 2025-2026 Integrallis Software, LLC. Apache-2.0.

output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.bench.id
}

output "public_dns" {
  description = "Public DNS — SSH in to watch /var/log/vectors-bench.log."
  value       = aws_instance.bench.public_dns
}

output "ssh_command" {
  description = "Convenience SSH command (tail the benchmark log)."
  value       = "ssh ubuntu@${aws_instance.bench.public_dns} 'tail -f /var/log/vectors-bench.log'"
}

output "results_uri" {
  description = "Where results are uploaded (empty if S3 upload is disabled)."
  value       = var.s3_results_uri
}
