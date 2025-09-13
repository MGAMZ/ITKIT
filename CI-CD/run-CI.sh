#!/usr/bin/env bash
set -euo pipefail

# 要测试的 Python 版本（你可以自由调整）
PY_VERSIONS=(3.10 3.11 3.12 3.13 3.14rc2)
DEBIAN_DISTRO="trixie"

# pytest 额外参数
PYTEST_ARGS="-q -k 'not gui'"

# 可选：是否安装 GUI 依赖并运行 gui 测试（默认关闭）
RUN_GUI_TESTS=false

# 显示帮助
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [run-gui: true|false]"
  exit 0
fi
if [[ "${1:-}" == "true" ]]; then
  RUN_GUI_TESTS=true
fi

REPO_DIR="$(pwd)"

for ver in "${PY_VERSIONS[@]}"; do
  docker_image="python:$ver-$DEBIAN_DISTRO-ITKIT"
  echo "=== Testing on docker image: $docker_image ==="

  docker run --rm -t \
    -v "$REPO_DIR":/src -w /src \
    $docker_image bash -eux -o pipefail -c "
      apt-get update -yq && apt-get install -yq --no-install-recommends \
        build-essential git ca-certificates wget python3-dev \
        # minimal libs that help Qt/SimpleITK wheels (optional)
        libxcb1 libx11-6 libxkbcommon0 libfontconfig1 libgl1 fonts-dejavu-core || true

      python -m pip install --upgrade pip setuptools wheel

      # Install package. If you want optional extras (gui), change accordingly:
      if [ \"$RUN_GUI_TESTS\" = true ]; then
        python -m pip install -e .[gui]
      else
        python -m pip install -e .
      fi

      # Install test deps if any (pytest already expected)
      python -m pip install pytest

      # Ensure QT offscreen for headless GUI tests (if enabled)
      export QT_QPA_PLATFORM=offscreen

      # Run pytest; skip gui tests by default
      python -m pytest $PYTEST_ARGS
    "
  echo "=== python:$ver done ==="
done

echo "All done."