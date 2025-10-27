#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname $(dirname $(realpath $BASH_SOURCE)))"
source "$path_repo/env.sh"

port=8889

show_help() {
    echo "Usage:"
    echo "  ./start_jupyter.sh [-h|--help] [-p|--port <port>]"
    echo
    echo "Start a jupyter server."
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
        -p | --port)
            port="$1"
            shift
            ;;
        *)
            echo "Unknown option $arg"
            exit 1
            ;;
        esac
    done
}

start_tmux_jupyter() {
    tmux new -s jupyter jupyter notebook --port "$port" --no-browser
}

main() {
    parse_args "$@"
    start_tmux_jupyter
}

main "$@"
