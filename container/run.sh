#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/config.sh"

readonly name_container="$NAME_CONTAINER_PYTORCH_PROJECT"
command=""
use_detach=1

show_help() {
    echo "Usage:"
    echo "  ./run.sh [-h|--help] [-a|--use_attach] [<command>]"
    echo
    echo "Run the container."
    echo
}

parse_args() {
    while [[ "$#" -gt 0 ]]; do
        local arg="$1"
        shift
        case "$arg" in
        -h | --help)
            show_help
            exit 0
            ;;
        -a | --use_attach)
            use_detach=""
            ;;
        *)
            if [[ -z "$command" ]]; then
                command="$arg"
            else
                command="$command $arg"
            fi
            ;;
        esac
    done
}

run() {
    local name_tag="$(arch)"
    local name_repo="$(basename "$path_repo")"

    docker run \
        --name "$name_container" \
        --interactive \
        --tty \
        --ipc host \
        --net host \
        --gpus all \
        --shm-size 64mb \
        --rm \
        --env DISPLAY \
        ${use_detach:+"--detach"} \
        --volume /etc/localtime:/etc/localtime:ro \
        --volume /tmp/.X11-unix/:/tmp/.X11-unix/:ro \
        --volume "$HOME/.Xauthority:/root/.Xauthority:ro" \
        --volume "$path_repo:/root/repos/$name_repo" \
        "$name_container:$name_tag" \
        ${command:+"$command"}
}

main() {
    parse_args "$@"
    run
}

main "$@"
