FROM nvcr.io/nvidia/pytorch:25.09-py3

# 设置非交互模式，避免apt安装时的用户输入提示
ENV DEBIAN_FRONTEND=noninteractive

# 步骤1: 配置APT源为阿里源
RUN echo 'Types: deb' > /etc/apt/sources.list.d/ali.sources && \
    echo 'URIs: https://mirrors.aliyun.com/ubuntu/' >> /etc/apt/sources.list.d/ali.sources && \
    echo 'Suites: noble noble-security noble-updates noble-backports' >> /etc/apt/sources.list.d/ali.sources && \
    echo 'Components: main restricted universe multiverse' >> /etc/apt/sources.list.d/ali.sources && \
    echo '' >> /etc/apt/sources.list.d/ali.sources && \
    echo 'Types: deb-src' >> /etc/apt/sources.list.d/ali.sources && \
    echo 'URIs: https://mirrors.aliyun.com/ubuntu/' >> /etc/apt/sources.list.d/ali.sources && \
    echo 'Suites: noble noble-security noble-updates noble-backports' >> /etc/apt/sources.list.d/ali.sources && \
    echo 'Components: main restricted universe multiverse' >> /etc/apt/sources.list.d/ali.sources && \
    rm -rf /etc/apt/sources.list.d/ubuntu.sources

# 步骤2 & 3: 更新APT包列表并进行完整升级
RUN apt update && \
    apt full-upgrade -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# 步骤4: 配置pip源为阿里源
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    echo "[install]" >> /root/.pip/pip.conf && \
    echo "trusted-host = mirrors.aliyun.com" >> /root/.pip/pip.conf

WORKDIR /workspace

# 步骤5: 克隆ITKIT仓库并安装
RUN git clone https://github.com/MGAMZ/ITKIT && \
    cd ITKIT && \
    pip install -e ".[dev,medical_vision,pytorch,mm,gui]"

CMD ["/bin/bash"]
