#!/usr/bin/env bash
set -euo pipefail

# Build ITKIT images for multiple Python trixie tags using tools/Dockerfile.itkit
TAGS=(
  "3.9-trixie"
  "3.10-trixie"
  "3.11-trixie"
  "3.12-trixie"
  "3.13-trixie"
  "3.14rc2-trixie"
)

ITKIT_REF=${ITKIT_REF:-auto}
# Auto-detect Dockerfile path if not provided
if [[ -z "${DOCKERFILE:-}" ]]; then
  if [[ -f CI-CD/Dockerfile.itkit ]]; then
    DOCKERFILE=CI-CD/Dockerfile.itkit
  elif [[ -f tools/Dockerfile.itkit ]]; then
    DOCKERFILE=tools/Dockerfile.itkit
  else
    echo "ERROR: Dockerfile.itkit not found. Set DOCKERFILE env var or place it under CI-CD/ or tools/." >&2
    exit 1
  fi
fi

for tag in "${TAGS[@]}"; do
  echo "==> Building itkit:${tag} from python:${tag} (ITKIT_REF=${ITKIT_REF})"
  docker build \
    --build-arg PYTHON_TAG="${tag}" \
    --build-arg ITKIT_REF="${ITKIT_REF}" \
    -t "itkit:${tag}" \
    -f "${DOCKERFILE}" .
  echo "==> Done: itkit:${tag}"
  echo
done

echo "All images built. Try: docker run -it --rm itkit:3.12-trixie bash"
