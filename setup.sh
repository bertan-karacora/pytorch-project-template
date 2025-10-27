#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname "$(realpath "$BASH_SOURCE")")"
source "$path_repo/env.sh"

readonly url_keyring_nvidia="$URL_KEYRING_NVIDIA_LOCAL"
readonly version_cuda="$VERSION_CUDA_LOCAL"
readonly version_cudnn="$VERSION_CUDNN_LOCAL"
readonly version_python="$VERSION_PYTHON_LOCAL"
readonly version_pip="$VERSION_PIP_LOCAL"
readonly version_setuptools="$VERSION_SETUPTOOLS_LOCAL"
readonly version_wheel="$VERSION_WHEEL_LOCAL"

show_help() {
    echo "Usage:"
    echo "  ./setup.sh [-h|--help]"
    echo
    echo "Setup system."
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
        *)
            echo "Unknown option $arg"
            exit 1
            ;;
        esac
    done
}

setup_system() {
    echo "Setting up system..."

    # Install essentials
    sudo apt-get update --quiet &&
        sudo apt-get install --quiet --assume-yes --no-install-recommends \
            apt-utils \
            build-essential \
            ca-certificates \
            curl \
            git \
            gnupg \
            htop \
            lsb-release \
            nano \
            software-properties-common \
            tmux \
            udev \
            unzip \
            wget \
            x11-apps

    # Install CUDA Toolkit
    wget "$url_keyring_nvidia" --quiet --directory-prefix /tmp && \
        sudo dpkg --install /tmp/cuda-keyring_1.1-1_all.deb && \
        rm /tmp/cuda*.deb
    sudo apt-get update --quiet && \
        sudo apt-get install --quiet --assume-yes --no-install-recommends \
            "cuda-toolkit-$version_cuda"

    # Install cuDNN
    sudo apt-get update --quiet && \
        sudo apt-get install --quiet --assume-yes --no-install-recommends \
            "cudnn$version_cudnn"

    echo "Setting up system finished"
}

setup_python() {
    echo "Setting up python ..."

    # Install Python
    sudo apt-get update --quiet &&
        sudo apt-get install --quiet --assume-yes --no-install-recommends \
            python-is-python3 \
            python3 \
            python3-pip \
            python3-venv

    # Install Python version explicitly
    sudo add-apt-repository --quiet --assume-yes ppa:deadsnakes/ppa
    sudo apt-get update --quiet &&
        sudo apt-get install --quiet --assume-yes --no-install-recommends \
            "python$version_python" \
            "python$version_python-venv"

    # Setup venv
    "python$version_python" -m venv "$path_repo/.venv"
    source "$path_repo/.venv/bin/activate"
    pip install --no-cache-dir \
        pip==$version_pip \
        setuptools==$version_setuptools \
        wheel==$version_wheel

    echo "Setting up python finished"
}

setup_libs() {
    echo "Setting up libraries ..."
    
    pip install --requirement "$path_repo/requirements.txt"

    echo "Setting up libraries finished"
}

main() {
    parse_args "$@"
    setup_system
    setup_python
    setup_libs
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
