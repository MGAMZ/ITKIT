FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN cat <<EOF > /etc/apt/sources.list.d/tsinghua.sources
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://security.ubuntu.com/ubuntu/
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF

RUN apt update
RUN apt install -y git build-essential
RUN apt upgrade -y

RUN git clone -b 'dev/main' --recursive https://github.com/MGAMZ/ITKIT.git

RUN pip install ./ITKIT[dev,medical_vision,pytorch,mm]

CMD ["bash"]