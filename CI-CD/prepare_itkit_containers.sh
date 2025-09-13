#!/usr/bin/env bash
set -euo pipefail

# Start ephemeral containers from python:<tag> and provision ITKIT inside each
TAGS=(
  "3.9-trixie"
  "3.10-trixie"
  "3.11-trixie"
  "3.12-trixie"
  "3.13-trixie"
  "3.14rc2-trixie"
)

for tag in "${TAGS[@]}"; do
  name="itkit-${tag//:/-}"
  echo "==> Preparing container ${name} from python:${tag}"
  docker pull "python:${tag}"
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d --name "${name}" --hostname "${name}" --workdir /workspace "python:${tag}" sleep infinity

  docker exec -i "${name}" bash -lc 'set -euo pipefail
# Configure Debian deb822 sources
cat >/etc/apt/sources.list.d/debian.sources << "EOF"
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/debian
Suites: trixie trixie-updates trixie-backports
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

# Types: deb-src
# URIs: https://mirrors.tuna.tsinghua.edu.cn/debian
# Suites: trixie trixie-updates trixie-backports
# Components: main contrib non-free non-free-firmware
# Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: https://security.debian.org/debian-security
Suites: trixie-security
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

# Types: deb-src
# URIs: https://security.debian.org/debian-security
# Suites: trixie-security
# Components: main contrib non-free non-free-firmware
# Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
update-ca-certificates || true

# Configure pip mirror
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Workspace and ITKIT
mkdir -p /workspace
cd /workspace
if [ ! -d ITKIT ]; then
  git clone https://gitee.com/MGAM/ITKIT.git
fi
python -m pip install --upgrade pip
pip install ./ITKIT[dev,medical_vision,mm,gui]

apt-get clean
rm -rf /var/lib/apt/lists/*
'
  echo "==> Container ${name} is ready"
  echo

done

echo "Use: docker exec -it itkit-3.12-trixie bash"
