#!/usr/bin/env bash

set -e -u -o pipefail

readonly path_repo="$(dirname $(dirname $(realpath $BASH_SOURCE)))"
source "$path_repo/env.sh"

readonly name_container="$NAME_CONTAINER_PYTORCH_PROJECT"
command=""
use_detach=""
gb_ram_system=""

show_help() {
    echo "Usage:"
    echo "  ./run.sh [-h|--help] [-a|--use_detach] [<command>]"
    echo
    echo "Run a command in the container."
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
        -a | --use_attach)
            use_detach=1
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

check_memory_system() {
    local kb_ram_system=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    gb_ram_system="$((kb_ram_system / 1024 / 1024))"

    if [ "$gb_ram_system" -lt 8 ]; then
        echo "Warning: System has less than 8GB of RAM (${gb_ram_system}GB)."
        echo "         The container may not work properly."
    fi
}

run() {
    local name_repo="$(basename "$path_repo")"
    local name_tag="$(arch)"

    docker run \
        --name "$name_container" \
        --shm-size "${gb_ram_system}g" \
        --gpus all \
        --ipc host \
        --interactive \
        --tty \
        --net host \
        --rm \
        --env DISPLAY \
        ${use_detach:+"--detach"} \
        --volume /etc/localtime:/etc/localtime:ro \
        --volume /tmp/.X11-unix/:/tmp/.X11-unix/:ro \
        --volume "$HOME/.Xauthority:/home/$USER/.Xauthority:ro" \
        --volume "$path_repo:/home/$USER/repos/$name_repo" \
        "$name_container:$name_tag" \
        ${command:+"$command"}
}

main() {
    parse_args "$@"
    check_memory_system
    run
}

main "$@"
