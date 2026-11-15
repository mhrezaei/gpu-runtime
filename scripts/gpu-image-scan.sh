#!/usr/bin/env bash
# Secret scan of a text dump (docker history, a file list, ...) for the GPU runtime image (#635, T11.16).
#   bash scripts/gpu-image-scan.sh <file|->   exit 0 clean, 1 finding (matching lines on stderr, values cut), 2 usage
# Kept apart from scripts/gpu-image.sh so it can be tested without Docker.
set -euo pipefail

# VALUES are secret values; PATTERN adds the bare name HF_TOKEN, which must never appear in a build command (docker history).
# The image filesystem is scanned for VALUES only: upstream code (ai-toolkit) reads HF_TOKEN from the env by name.
VALUES='hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE|AKIA[0-9A-Z]{16}|X-Amz-Signature'
PATTERN="HF_TOKEN|${VALUES}"
if [ "${1:-}" = "--pattern" ]; then printf '%s\n' "$PATTERN"; exit 0; fi
if [ "${1:-}" = "--value-pattern" ]; then printf '%s\n' "$VALUES"; exit 0; fi

SRC="${1:--}"
if [ "$SRC" = "-" ]; then DUMP="$(cat)"; elif [ -f "$SRC" ]; then DUMP="$(cat "$SRC")"; else echo "usage: gpu-image-scan.sh <file|->" >&2; exit 2; fi

# print only the first 60 characters of a finding so the log never carries a whole secret
if HITS="$(printf '%s\n' "$DUMP" | grep -E "$PATTERN" | cut -c1-60)"; then
  printf '%s\n' "$HITS" >&2
  echo "FINDING: a secret-looking string is in the scanned text" >&2
  exit 1
fi
exit 0
