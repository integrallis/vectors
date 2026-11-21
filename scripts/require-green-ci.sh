#!/usr/bin/env bash
# Refuses a release unless every GitHub Actions workflow is green.
#
#   scripts/require-green-ci.sh <owner/repo> <branch> <sha> [<this-run-id>]
#
# Two conditions, both required:
#   1. Every workflow run for <sha> (other than the calling run) has finished and succeeded, and
#      at least one such run exists. In-flight runs are waited for, up to WAIT_MINUTES (default 45).
#   2. For every active workflow, its most recent run on <branch> succeeded. This is the condition
#      that catches a path-filtered or scheduled workflow left red by an EARLIER commit: 0.1.44 was
#      published while Framework Compatibility's last run on main was a failure, because that
#      workflow did not run on the release commit at all.
#
# The workflow named by SELF_WORKFLOW (default "Release") is excluded from both, since it is the
# one asking. Needs `gh` and a token with actions:read. Exit 0 = green, 1 = not green.
set -euo pipefail

REPO="${1:?owner/repo}"
BRANCH="${2:?branch}"
SHA="${3:?sha}"
SELF_RUN="${4:-0}"
SELF_WORKFLOW="${SELF_WORKFLOW:-Release}"
WAIT_MINUTES="${WAIT_MINUTES:-45}"

runs_for_sha() {
  gh api --paginate "repos/$REPO/actions/runs?head_sha=$SHA&per_page=100" \
    --jq ".workflow_runs[] | select(.id != $SELF_RUN and .name != \"$SELF_WORKFLOW\")
          | [.name, .status, (.conclusion // \"-\"), .html_url] | @tsv"
}

deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
while :; do
  sha_runs="$(runs_for_sha)"
  pending="$(printf '%s\n' "$sha_runs" | awk -F'\t' 'NF && $2 != "completed"')"
  [[ -z "$pending" ]] && break
  if (( $(date +%s) >= deadline )); then
    echo "::error::Workflows for $SHA were still running after ${WAIT_MINUTES}m:"
    printf '%s\n' "$pending"
    exit 1
  fi
  echo "Waiting for $(printf '%s\n' "$pending" | wc -l | tr -d ' ') workflow run(s) on $SHA ..."
  sleep 30
done

failed=0

if [[ -z "$sha_runs" ]]; then
  echo "::error::No workflow has run for $SHA. A commit that CI never built is not released."
  failed=1
fi
while IFS=$'\t' read -r name status conclusion url; do
  [[ -z "$name" ]] && continue
  if [[ "$conclusion" != "success" ]]; then
    echo "::error::[$SHA] $name: $conclusion  $url"
    failed=1
  else
    echo "ok   [commit] $name"
  fi
done <<< "$sha_runs"

while IFS=$'\t' read -r wf_id wf_name; do
  [[ -z "$wf_id" || "$wf_name" == "$SELF_WORKFLOW" ]] && continue
  latest="$(gh api "repos/$REPO/actions/workflows/$wf_id/runs?branch=$BRANCH&per_page=1" \
    --jq '.workflow_runs[0] // empty | [.status, (.conclusion // "-"), .head_sha[0:8], .html_url] | @tsv')"
  if [[ -z "$latest" ]]; then
    echo "n/a  [$BRANCH] $wf_name (never run on $BRANCH)"
    continue
  fi
  IFS=$'\t' read -r status conclusion short url <<< "$latest"
  if [[ "$status" != "completed" || "$conclusion" != "success" ]]; then
    echo "::error::[$BRANCH] $wf_name: latest run is $status/$conclusion at $short  $url"
    failed=1
  else
    echo "ok   [$BRANCH] $wf_name ($short)"
  fi
done < <(gh api --paginate "repos/$REPO/actions/workflows?per_page=100" \
           --jq '.workflows[] | select(.state == "active") | [.id, .name] | @tsv')

if (( failed )); then
  echo "::error::CI is not green. Refusing to release. Fix or re-run the red workflows first."
  exit 1
fi
echo "All workflows green for $SHA and on $BRANCH."
