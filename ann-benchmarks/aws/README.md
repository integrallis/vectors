# Running the `vectors` ANN-Benchmarks on AWS

The ANN-Benchmarks suite is **single-node and single-threaded by design** — that
is what makes recall-vs-QPS numbers comparable across systems. So the right infra
is **one fixed EC2 instance + a script**, not Kubernetes and not autoscaling. Two
ways to use what's here:

## Option 1 — Terraform (provision + run + collect)

```bash
cd ann-benchmarks/aws
terraform init
terraform apply \
  -var 'key_name=my-keypair' \
  -var 'ssh_cidr=203.0.113.4/32' \
  -var 'instance_type=c7i.4xlarge' \
  -var 'dataset=sift-128-euclidean' \
  -var 'algorithms=vectors hnswlib faiss' \
  -var 's3_results_uri=s3://my-bucket/vectors-bench' \
  -var 's3_results_bucket=my-bucket'

# Watch progress:
ssh ubuntu@$(terraform output -raw public_dns) 'tail -f /var/log/vectors-bench.log'

# When done, results are in s3://my-bucket/vectors-bench/ and on the box under
# /opt/vectors-bench/ann-benchmarks/results/. Then:
terraform destroy
```

`main.tf` launches one Ubuntu 22.04 instance, a security group (SSH from your
CIDR), and — only when `s3_results_bucket` is set — an instance role allowed to
`PutObject` to that bucket. `user_data` clones the repo and runs
`run-benchmark.sh`.

## Option 2 — the script alone (any Ubuntu 22.04 host)

`run-benchmark.sh` has no AWS dependency. Copy it to any box (an EC2 instance you
already manage, a bare-metal server, …) and run it:

```bash
DATASET=glove-100-angular \
ALGORITHMS="vectors vectors-jpype hnswlib" \
S3_RESULTS=s3://my-bucket/vectors-bench \
  ./run-benchmark.sh
```

It installs Docker + git + Temurin JDK 25, builds `vectors-server`, stages both
adapters (HTTP `vectors` and in-process `vectors-jpype`) into an ann-benchmarks
checkout, runs `install.py` / `run.py` / `plot.py` for the requested algorithms,
and uploads `results/` to S3 if `S3_RESULTS` is set. See the script header for all
env vars.

## Getting a head-to-head

Set `ALGORITHMS` to `vectors` plus the competitors you care about — e.g.
`"vectors hnswlib faiss qdrant pgvector"`. The harness builds each in its own
container and `plot.py` overlays every curve on one recall-vs-QPS chart. Because
all of them run on the **same instance in the same harness**, the comparison is
true apples-to-apples.

## Notes / caveats

- Pick an `instance_type` and **keep it constant** across runs for reproducibility.
  Size it so the dataset + indexes fit in RAM; `c7i.4xlarge` / `r7i.4xlarge` are
  reasonable starting points for the million-scale datasets.
- This Terraform is a **starting template** — it assumes a default VPC with public
  subnets and internet egress. Adjust the AMI owner/filter, subnet, and IAM to
  your account's conventions.
- The instance keeps running until you `terraform destroy` (or shut it down). A
  full multi-algorithm sweep is hours of compute — budget accordingly.
