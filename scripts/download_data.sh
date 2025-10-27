#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname $(dirname $(realpath $BASH_SOURCE)))"
source "$path_repo/env.sh"

path_dir_data=""
readonly urls=()

show_help() {
    echo "Usage:"
    echo "  ./download_data.sh <path_dir_data>"
    echo
    echo "Download data to <path_dir_data>."
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
            if [[ -z "$path_dir_data" ]]; then
                path_dir_data="$arg"
            else
                echo "Unknown option $arg"
                exit 1
            fi
            ;;
        esac
    done
}

download_data() {
    if [ ! -d "$path_dir_data" ]; then
        mkdir --parents "$path_dir_data"
    fi

    for url in "${urls[@]}"; do
        local name_file="$(basename "$url")"
        local path="$path_dir_data/$name_file"

        if [ ! -f "$path" ]; then
            echo "Downloading $name_file to $path_dir_data ..."
            wget "$url" --directory-prefix "$path_dir_data" --quiet --show-progress
        else
            echo "$name_file already exists in $path_dir_data, skipping download"
        fi
    done
}

main() {
    parse_args "$@"
    download_data
}

main "$@"