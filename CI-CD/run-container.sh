#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-3.12-trixie}
NAME=${NAME:-itkit-${TAG//:/-}}
MOUNT_REPO=${MOUNT_REPO:-true}
EXTRA_DOCKER_ARGS=${EXTRA_DOCKER_ARGS:-}

IMAGE="itkit:${TAG}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found. Build it first with tools/build-images.sh" >&2
  exit 1
fi

DOCKER_ARGS=( -it --rm --name "$NAME" -w /workspace )
if [[ "$MOUNT_REPO" == "true" ]]; then
  DOCKER_ARGS+=( -v "$PWD":/workspace/hostrepo )
fi

exec docker run "${DOCKER_ARGS[@]}" $EXTRA_DOCKER_ARGS "$IMAGE" bash
