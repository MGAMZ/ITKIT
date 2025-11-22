set -euo pipefail

# 要测试的 Python 版本
PY_VERSIONS=(3.10 3.11 3.12 3.13)
DEBIAN_DISTRO="trixie"
PYTEST_ARGS="-q"
REPO_DIR="$(pwd)"

# 解析 ITKIT 版本号（简化：单次 awk，若失败则由 set -e 直接失败）
ITKIT_VERSION=${ITKIT_VERSION:-}
if [[ -z "${ITKIT_VERSION}" ]]; then
  ITKIT_VERSION=$(awk '
    BEGIN{inproj=0}
    /^\[project\]/{inproj=1; next}
    /^\[/{if(inproj){exit}; next}
    inproj && $1 ~ /^version/ {
      sub(/version[[:space:]]*=[[:space:]]*"/, "", $0);
      sub(/".*/, "", $0);
      print $0; exit
    }
  ' pyproject.toml)
fi

for ver in "${PY_VERSIONS[@]}"; do
  image_tag="itkit:${ITKIT_VERSION}-${ver}-${DEBIAN_DISTRO}"
  echo "=== Testing on docker image: ${image_tag} ==="

  docker run --rm -t \
    -e QT_QPA_PLATFORM=offscreen \
    -v "${REPO_DIR}":/src -w /src \
    "${image_tag}" bash -eux -o pipefail -c "
      python -m pytest ${PYTEST_ARGS}
    "

  echo "=== python:${ver} done ==="
done

echo "All done."
