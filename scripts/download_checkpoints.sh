#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname $(dirname $(realpath $BASH_SOURCE)))"
source "$path_repo/env.sh"

path_dir_checkpoints=""
readonly urls=()

show_help() {
    echo "Usage:"
    echo "  ./download_checkpoints.sh <path_dir_checkpoints>"
    echo
    echo "Download checkpoints to <path_dir_checkpoints>."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case "$arg" in
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$path_dir_checkpoints" ]]; then
                path_dir_checkpoints="$arg"
            else
                echo "Unknown option $arg"
                exit 1
            fi
            ;;
        esac
    done
}

download_checkpoints() {
    if [ ! -d "$path_dir_checkpoints" ]; then
        mkdir --parents "$path_dir_checkpoints"
    fi

    for url in "${urls[@]}"; do
        local name_file="$(basename "$url")"
        local path="$path_dir_checkpoints/$name_file"

        if [ ! -f "$path" ]; then
            echo "Downloading $name_file to $path_dir_checkpoints ..."
            wget "$url" --directory-prefix "$path_dir_checkpoints" --quiet --show-progress
        else
            echo "$name_file already exists in $path_dir_checkpoints, skipping download"
        fi
    done
}

main() {
    parse_args "$@"
    download_checkpoints
}

main "$@"