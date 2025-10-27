#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname $(dirname $(realpath $BASH_SOURCE)))"
source "$path_repo/env.sh"

readonly name_container="$NAME_CONTAINER_PYTORCH_PROJECT"
use_clean=""
use_debug=""

show_help() {
    echo "Usage:"
    echo "  ./build.sh [-h|--help] [--use_clean] [--use_debug]"
    echo
    echo "Build the container image."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case $arg in
        -h | --help)
            show_help
            exit 0
            ;;
        --use_clean)
            use_clean=1
            ;;
        --use_debug)
            use_debug=1
            ;;
        *)
            echo "Unknown option $arg"
            exit 1
            ;;
        esac
    done
}

build() {
    local name_tag="$(arch)"

    docker build \
        --build-arg=USER \
        --build-arg UID="$UID" \
        --build-arg=VERSION_UBUNTU_CONTAINER \
        --build-arg=URL_KEYRING_NVIDIA_CONTAINER \
        --build-arg=VERSION_CUDA_CONTAINER \
        --build-arg=VERSION_CUDNN_CONTAINER \
        --build-arg=VERSION_PYTHON_CONTAINER \
        --build-arg=VERSION_PIP_CONTAINER \
        --build-arg=VERSION_SETUPTOOLS_CONTAINER \
        --build-arg=VERSION_WHEEL_CONTAINER \
        --tag="$name_container:$name_tag" \
        --file="$path_repo/container/Dockerfile" \
        ${use_clean:+--no-cache} \
        ${use_debug:+--progress=plain} \
        "$path_repo"
}

main() {
    parse_args "$@"
    build
}

main "$@"
