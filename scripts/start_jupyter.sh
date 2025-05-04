#!/usr/bin/env bash

set -e -u -x -o pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/config.sh"

show_help() {
    echo "Usage:"
    echo "  ./start_jupyter.sh [-h|--help]"
    echo
    echo "Start a jupyter server."
    echo
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        local arg="$1"
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

start_tmux_jupyter() {
    tmux new -s jupyter jupyter notebook --no-browser --allow-root --port 8999
}

main() {
    parse_args "$@"
    start_tmux_jupyter
}

main "$@"
