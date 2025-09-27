# How to build ITKIT docker image

## Script Function

### Dockerfile.itkit

A Dockerfile to build ITKIT docker image based on python trixie image, it adds the following features:

- Configure `apt` mirror server.
- Install system dependencies.
- Configure `pip` mirror server.
- Clone ITKIT source code from GitHub, `ITKIT_REF` specify the branch name.
- Install python dependencies and `ITKIT` package.

### build_docker_itkit.sh

Build ITKIT docker image using `Dockerfile.itkit`. It can generate multiple images with different `python` versions.

### run-CI.sh

Run ITKIT docker image interactively, it mounts the current directory to `/workspace/ITKIT` in the container.

## How to use

1. Run `build_docker_itkit.sh` to build ITKIT docker image.
2. Run `run-CI.sh` to start a container interactively.
