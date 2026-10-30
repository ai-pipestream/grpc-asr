#!/bin/sh
# Exercise publication gates with a fake Docker CLI; never contacts a registry.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/bin"
export TEST_CALLS="$scratch/calls"
export TEST_ARM="pipestreamai/whisper-grpc@sha256:$(printf '%064d' 2)"
amd="pipestreamai/whisper-grpc@sha256:$(printf '%064d' 1)"
commit=0123456789012345678901234567890123456789
printf '%s\n' "$amd" > "$scratch/amd"
printf '%s\n' "$TEST_ARM" > "$scratch/arm"
cat > "$scratch/bin/docker" <<'MOCK'
#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$TEST_CALLS"
if [ "$1" = pull ]; then exit 0; fi
if [ "$1" = image ] && [ "$2" = inspect ]; then
  if [ "$5" = "$TEST_ARM" ] && [ "${WRONG_ARM:-0}" = 0 ]; then echo linux/arm64; else echo linux/amd64; fi
  exit 0
fi
if [ "$1 $2 $3" = 'buildx imagetools create' ]; then exit 0; fi
if [ "$1 $2 $3" = 'buildx imagetools inspect' ]; then
  echo '{"manifests":[{"platform":{"os":"linux","architecture":"amd64"}},{"platform":{"os":"linux","architecture":"arm64"}},{"platform":{"os":"unknown","architecture":"unknown"}}]}'
  exit 0
fi
exit 1
MOCK
chmod +x "$scratch/bin/docker"
PATH="$scratch/bin:$PATH"
export PATH
sh "$root/scripts/publish-manifest.sh" "$commit" "$scratch/amd" "$scratch/arm" >/dev/null
grep -Fq "buildx imagetools create -t pipestreamai/whisper-grpc:sha-$commit -t pipestreamai/whisper-grpc:latest $amd $TEST_ARM" "$TEST_CALLS"
reject() {
  : > "$TEST_CALLS"
  if sh "$root/scripts/publish-manifest.sh" "$@" > "$scratch/error" 2>&1; then
    echo 'Invalid publication was accepted' >&2; exit 1
  fi
  if grep -q 'imagetools create' "$TEST_CALLS"; then
    echo 'Invalid publication moved a release tag' >&2; exit 1
  fi
}
reject bad-commit "$scratch/amd" "$scratch/arm"
reject "$commit" "$scratch/amd" "$scratch/missing"
reject "$commit" "$scratch/amd" "$scratch/amd"
printf '%s\n' 'another/repo@sha256:invalid' > "$scratch/bad"
reject "$commit" "$scratch/amd" "$scratch/bad"
printf '%s\n%s\n' "$amd" extra > "$scratch/multiline"
reject "$commit" "$scratch/multiline" "$scratch/arm"
export WRONG_ARM=1
reject "$commit" "$scratch/amd" "$scratch/arm"
printf 'Manifest publication: success and six rejection gates passed\n'
