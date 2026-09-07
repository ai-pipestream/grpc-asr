#!/usr/bin/env bash
# Turns the per-platform images a publish run pushed by digest into the
# tagged manifest lists the public names point at, then proves the result.
#
# The publish workflow builds every platform leg with provenance and SBOM
# attestations and pushes it by digest only (no tag), boot-proofs that
# digest, and hands the digests that passed to this script. Nothing is
# visible under a public tag until this script names it, so the smoke gate
# always runs before publication and the attestations built into each leg
# travel into the manifest list untouched.
#
# Usage:
#   publish-manifests.sh --suffix SUFFIX --version VERSION \
#       --platforms linux/amd64[,linux/arm64] \
#       --repo REPO [--repo REPO ...] DIGEST...
#
#   SUFFIX    the variant suffix, '' for the CUDA image, -cpu, -openvino
#   VERSION   the release version, or '' to publish only latest<SUFFIX>
#   REPO      a repository the digests were pushed to
#             (docker.io/pipestreamai/grpc-asr, ...); each repository gets
#             its own manifest list from its own copies, which is what
#             `docker buildx imagetools create` requires
#   DIGEST    one sha256 per platform, in any order
#
# For each repository it runs `docker buildx imagetools create` with every
# tag the run owns (latest<SUFFIX>, and <VERSION><SUFFIX> when a version is
# given) over REPO@DIGEST sources, then inspects the published index and
# fails unless it lists exactly the requested platforms and carries one
# attestation manifest per platform. The docker binary comes from PATH.
set -euo pipefail

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -n -1 >&2
  exit 64
}

suffix=
version=
platforms=
repos=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suffix) [[ $# -ge 2 ]] || usage; suffix=$2; shift 2 ;;
    --version) [[ $# -ge 2 ]] || usage; version=$2; shift 2 ;;
    --platforms) [[ $# -ge 2 ]] || usage; platforms=$2; shift 2 ;;
    --repo) [[ $# -ge 2 ]] || usage; repos+=("$2"); shift 2 ;;
    --) shift; break ;;
    -*) usage ;;
    *) break ;;
  esac
done
[[ ${#repos[@]} -gt 0 && -n "$platforms" ]] || usage

# Every digest is normalised to the sha256:<hex> form the registry reference
# needs, whether it arrived that way or as the bare hex an artifact filename
# carries.
digests=()
for raw in "$@"; do
  hex=${raw#sha256:}
  if [[ ! "$hex" =~ ^[0-9a-f]{64}$ ]]; then
    echo "not a sha256 digest: '$raw'" >&2
    exit 1
  fi
  digests+=("sha256:$hex")
done

IFS=',' read -r -a expected_platforms <<<"$platforms"
if [[ ${#digests[@]} -ne ${#expected_platforms[@]} ]]; then
  echo "expected ${#expected_platforms[@]} digest(s) for platforms $platforms, got ${#digests[@]}:" >&2
  echo "a platform leg did not pass its build and smoke gate, so nothing is tagged" >&2
  exit 1
fi

tags_for_run() {
  echo "latest${suffix}"
  [[ -n "$version" ]] && echo "${version}${suffix}"
  return 0
}

create_manifest_list() {
  local repo=$1 tag args=() sources=()
  while read -r tag; do
    args+=(-t "${repo}:${tag}")
  done < <(tags_for_run)
  local digest
  for digest in "${digests[@]}"; do
    sources+=("${repo}@${digest}")
  done
  echo "== publish: ${args[*]} <- ${sources[*]}"
  docker buildx imagetools create "${args[@]}" "${sources[@]}"
}

# The platforms an index lists, one per line, attestation manifests
# excluded (BuildKit marks those unknown/unknown).
index_platforms() {
  jq -r '.manifests[] | select(.platform.os != "unknown")
         | "\(.platform.os)/\(.platform.architecture)"' | sort -u
}

attestation_count() {
  jq -r '[.manifests[] | select(.platform.os == "unknown")] | length'
}

verify_tag() {
  local ref=$1 index actual expected
  index=$(docker buildx imagetools inspect --raw "$ref")
  actual=$(index_platforms <<<"$index")
  expected=$(printf '%s\n' "${expected_platforms[@]}" | sort -u)
  if [[ "$actual" != "$expected" ]]; then
    echo "$ref lists platforms [$(tr '\n' ' ' <<<"$actual")] but the run built [$(tr '\n' ' ' <<<"$expected")]" >&2
    exit 1
  fi
  local attestations
  attestations=$(attestation_count <<<"$index")
  if [[ "$attestations" -ne ${#expected_platforms[@]} ]]; then
    echo "$ref carries $attestations attestation manifest(s) for ${#expected_platforms[@]} platform(s); provenance/SBOM did not attach" >&2
    exit 1
  fi
  echo "== publish: $ref ok, platforms [$(tr '\n' ' ' <<<"$actual")], $attestations attestation manifest(s)"
}

for repo in "${repos[@]}"; do
  create_manifest_list "$repo"
  while read -r tag; do
    verify_tag "${repo}:${tag}"
  done < <(tags_for_run)
done
