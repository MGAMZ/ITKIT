# CI-CD for ITKIT Containers

本目录提供两种方式来准备 ITKIT 环境：

- 基于 Dockerfile 预构建镜像（推荐，具备缓存与可重复性）
- 直接启动官方 `python:<tag>-trixie` 容器并在其中配置（一次性/调试友好）

目标：在 Debian trixie 的 Python 3.9 ~ 3.14rc2 系列下，统一设置 apt 源（清华 + security）、pip 源（清华），拉取并安装 ITKIT 及可选组件 `[dev,medical_vision,mm,gui]`。

## 文件清单

- `Dockerfile.itkit`：参数化 Dockerfile，通过 `--build-arg PYTHON_TAG=<ver>-trixie` 选择基础镜像。
- `build-images.sh`：批量构建 `itkit:<tag>` 镜像。默认 tags：`3.9-trixie .. 3.14rc2-trixie`。
- `run-container.sh`：使用已构建的 `itkit:<tag>` 镜像启动交互容器（可选挂载当前仓库）。
- `prepare_itkit_containers.sh`：直接以官方 `python:<tag>` 起容器，在容器内配置 apt/pip/ITKIT。
- `run-CI.sh`：示例 CI 运行脚本，演示在若干 Python 版本镜像中运行 pytest（默认跳过 GUI 测试）。

> 注意：如果 `Dockerfile.itkit` 位于本目录（CI-CD）下，请将构建脚本或命令中的 `-f` 路径替换为 `CI-CD/Dockerfile.itkit`。

## 预构建镜像（推荐）

批量构建：

```bash
chmod +x CI-CD/build-images.sh
./CI-CD/build-images.sh
```

- 环境变量：
  - `ITKIT_REF`：默认 `auto`，自动使用远端默认分支；也可指定分支/标签/提交。
  - `DOCKERFILE`：默认 `tools/Dockerfile.itkit`。若 Dockerfile 位于本目录，运行前设置：

    ```bash
    export DOCKERFILE=CI-CD/Dockerfile.itkit
    ```

单镜像构建示例：

```bash
docker build \
  --build-arg PYTHON_TAG=3.12-trixie \
  --build-arg ITKIT_REF=auto \
  -t itkit:3.12-trixie \
  -f CI-CD/Dockerfile.itkit .
```

运行已构建镜像：

```bash
chmod +x CI-CD/run-container.sh
CI-CD/run-container.sh 3.12-trixie
```

- 变量：
  - `MOUNT_REPO=true|false`（默认 true）：是否将当前目录挂载到 `/workspace/hostrepo`。
  - `NAME`：容器名（默认 `itkit-<tag>`）。

## 一键准备运行容器（非构建）

直接使用官方 `python:<tag>` 启动容器并配置环境（适合一次性/调试）：

```bash
chmod +x CI-CD/prepare_itkit_containers.sh
./CI-CD/prepare_itkit_containers.sh
```

完成后进入某个容器：

```bash
docker exec -it itkit-3.12-trixie bash
```

## CI 示例运行

`run-CI.sh` 演示如何在不同 Python 版本镜像中运行 pytest：

```bash
chmod +x CI-CD/run-CI.sh
CI-CD/run-CI.sh         # 默认跳过 GUI 测试
CI-CD/run-CI.sh true    # 启用 GUI 测试（headless）
```

- 变量：
  - `PY_VERSIONS`：在脚本顶部调整待测版本。
  - `PYTEST_ARGS`：自定义 pytest 参数。

## 常见问题

- 3.14rc2 可能缺少部分三方包轮子，构建或安装时间可能较长，建议优先使用 3.12/3.13。
- 如需代理，请在构建/运行前导出 `HTTP_PROXY/HTTPS_PROXY/NO_PROXY`。
- 若希望使用本地代码而不是克隆：在运行容器时挂载 `-v "$PWD":/workspace/ITKIT`，并在 Dockerfile 中去掉 `git clone` 与 `pip install ./ITKIT[...]` 步骤。
