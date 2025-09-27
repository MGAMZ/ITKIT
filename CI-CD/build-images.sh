#!/usr/bin/env bash
set -euo pipefail

# Build ITKIT images for multiple Python trixie tags using tools/Dockerfile.itkit
TAGS=(
  "3.10-trixie"
  "3.11-trixie"
  "3.12-trixie"
  "3.13-trixie"
)

ITKIT_REF=${ITKIT_REF:-auto}
# Detect ITKIT version from pyproject.toml unless overridden
ITKIT_VERSION=${ITKIT_VERSION:-}
if [[ -z "$ITKIT_VERSION" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    set +e
    ITKIT_VERSION=$(python3 - <<'PY'
import sys
try:
    import tomllib  # py311+
except Exception:
    try:
        import tomli as tomllib  # fallback if installed
    except Exception:
        sys.exit(2)
with open('pyproject.toml','rb') as f:
    data = tomllib.load(f)
v = (data.get('project') or {}).get('version')
if not v:
    sys.exit(3)
print(v)
PY
    )
    py_status=$?
    set -e
    if [[ $py_status -ne 0 || -z "$ITKIT_VERSION" ]]; then
      # awk fallback within [project] table
      ITKIT_VERSION=$(awk '
        BEGIN{inproj=0}
        /^\[project\]/{inproj=1; next}
        /^\[/{if(inproj){exit}; next}
        inproj && $1 ~ /^version/ {
          sub(/version[[:space:]]*=[[:space:]]*"/, "", $0);
          sub(/".*/, "", $0);
          print $0; exit
        }
      ' pyproject.toml || true)
    fi
  else
    # awk fallback if no python3
    ITKIT_VERSION=$(awk '
      BEGIN{inproj=0}
      /^\[project\]/{inproj=1; next}
      /^\[/{if(inproj){exit}; next}
      inproj && $1 ~ /^version/ {
        sub(/version[[:space:]]*=[[:space:]]*"/, "", $0);
        sub(/".*/, "", $0);
        print $0; exit
      }
    ' pyproject.toml || true)
  fi
fi

DOCKERFILE=CI-CD/Dockerfile.itkit

for tag in "${TAGS[@]}"; do
  image_tag="itkit:${ITKIT_VERSION}-${tag}"
  echo "==> Building ${image_tag} from python:${tag} (ITKIT_REF=${ITKIT_REF})"
  docker build \
    --build-arg PYTHON_TAG="${tag}" \
    --build-arg ITKIT_REF="${ITKIT_REF}" \
    -t "py${image_tag}" \
    -f "${DOCKERFILE}" .
  echo "==> Done: ${image_tag}"
  echo
done
