# Deploying vectors on AWS (and Cloudflare R2)

Run the object-storage-backed vector database **in your own cloud account**, with the data on
**your own S3 (or R2) bucket**. This directory holds the reference deployment and the
one-click CloudFormation template.

## Reference architecture (single node)

- **EC2 with local NVMe** as the hot cache (`i4i` x86, or `im4gn` arm64). Losing the NVMe is never
  data loss — the durable copy is in object storage, so a replacement node just rebuilds the cache.
- **S3** as the durable floor (block-public-access on, SSE, versioned).
- **IAM instance role** for S3 access — **no static keys**; the server uses the AWS default
  credential chain (IMDSv2). Set only `VECTORS_S3_BUCKET` + `VECTORS_S3_REGION`.
- **Server + Studio** in Docker on the instance.
- **~$150/month** for the `i4i.large` reference (instance + a modest S3 bill).

For production, front it with an **ALB + ACM certificate** (free public TLS), split `/api` and
`/studio` with path rules, and restrict `AllowedCidr` — the single-node template exposes the ports
directly for a fast start.

## One-click: the Launch Stack URL

`vectors-launch-stack.yaml` provisions the S3 bucket + IAM role + EC2 (NVMe) + security group and
boots the stack. It resolves the Amazon Linux 2023 AMI at deploy time (no hardcoded AMI IDs) and
bakes **no** credentials.

**Prerequisite:** publish the `vectors-server` (and optionally `vectors-studio-web`) images to a
public registry (GHCR / Docker Hub / ECR Public) and pass them as `ServerImage` / `StudioImage`.

Host the template on a public S3 URL, then hand users a quick-create link — it opens the
CloudFormation console in *their* account, pre-filled, and nothing is created until they click
**Create stack**:

```
https://<region>.console.aws.amazon.com/cloudformation/home?region=<region>#/stacks/create/review\
?templateURL=https://<bucket>.s3.amazonaws.com/vectors-launch-stack.yaml\
&stackName=vectors\
&param_ServerImage=ghcr.io/integrallis/vectors-server:0.1.0
```

Or from the CLI:

```bash
aws cloudformation deploy --template-file vectors-launch-stack.yaml --stack-name vectors \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides ServerImage=ghcr.io/integrallis/vectors-server:0.1.0 BearerToken=<token>
```

## Go-to-market path

The market gap (from the competitive research): every object-storage vector DB is either a
managed SaaS that runs in the *vendor's* account, or a fragmented BYOC delivered via
Terraform/Pulumi/Helm — **none is truly one-click into the customer's own account**. A single-click
CloudFormation "Launch Stack" that stands up the data plane in the customer's VPC on their own S3
bucket is best-in-class turnkey UX, and *"your data never leaves your account"* is a differentiator
MongoDB Atlas structurally cannot claim.

- **Phase 0 — now:** the Launch Stack URL above (a "Deploy to AWS" button). No Marketplace approval.
- **Phase 1 — AWS Marketplace, AMI + CloudFormation product.** Reuse this template (swap the AMI to
  an SSM parameter). Register as a seller (tax + bank; ~2–4 week review). List as **BYOL + a paid
  hourly option** to get listed fast. Gains discovery, the Marketplace "Launch with CloudFormation"
  one-click, and **consolidated AWS billing that draws down the customer's committed spend (EDP)** —
  an enterprise lever most competitors under-market. (Server-product fee is 20%.)
- **Phase 2 — metering / contracts.** Integrate the AWS Marketplace Metering Service (`MeterUsage`,
  instance-role-signed, hourly) or annual/contract pricing for enterprise deals.
- **Skip:** pure SaaS (wrong data-residency model), AMI-only (undercuts the S3-native cost story),
  and EKS Marketplace Quick Launch (discontinued March 1, 2026 — use your own CloudFormation).

## Cloudflare R2 as the durable floor

The server is S3-compatible, so **R2 works as the durable floor** with an endpoint override — and
**R2 charges zero egress**, which turns S3's largest and least-predictable line item into $0 (on
the order of $1k–$90k+/year saved for read-heavy workloads), on top of $0.015/GB storage. There is
**no** Cloudflare one-click/marketplace channel for a JVM server (Workers can't run a JVM;
Containers have no persistent local disk for the cache tier), so on Cloudflare the pattern is
**R2 (BYOC) + the server on your own compute**, optionally fronted by **Cloudflare Tunnel + Access**
(free, no inbound ports). Point the server at R2 with:

```
VECTORS_S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
VECTORS_S3_BUCKET=<bucket>
VECTORS_S3_REGION=auto
VECTORS_S3_ACCESS_KEY=<r2-token-id>
VECTORS_S3_SECRET_KEY=<sha256-of-r2-token-value>
```

Cloudflare's own **Vectorize** is R2-backed too — it validates this architecture, while its
Worker-only coupling and 1536-dim / 10M-vector / topK≤100 caps are exactly the gaps a portable
JVM engine on any S3-compatible store fills.
