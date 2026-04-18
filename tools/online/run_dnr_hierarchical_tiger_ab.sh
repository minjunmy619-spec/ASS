#!/usr/bin/env bash

set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ROOT="recipes/dnr/models"
TRAIN_ENTRY="aiaccel/aiaccel/torch/apps/train.py"
LOG_ROOT_DEFAULT="logs/dnr_hierarchical_tiger_ab"
DRY_RUN=0
CONTINUE_ON_ERROR=1
LOG_ROOT="${LOG_ROOT_DEFAULT}"

usage() {
    cat <<EOF
Usage: $0 [options]

Runs the focused DnR hierarchical comparison:
  1. hierarchical-soft-band musical
  2. hierarchical-soft-band-ffi speech_lowfreq_narrow
  3. hierarchical-soft-band-parallel-ffi speech_lowfreq_narrow (rt192k)

options:
  --dry-run              Print the selected training commands without launching
  --stop-on-error        Stop on the first failed config
  --log-root <dir>       Directory for per-run log files
  -h, --help             Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --stop-on-error)
            CONTINUE_ON_ERROR=0
            shift
            ;;
        --log-root)
            if [ "$#" -lt 2 ]; then
                echo "--log-root requires a directory argument" >&2
                exit 1
            fi
            LOG_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_ROOT}"

if [ -z "${ngpu:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        ngpu="$(nvidia-smi -L | wc -l)"
    else
        ngpu=1
    fi
fi

if [ -z "${num_nodes:-}" ]; then
    if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
        num_nodes="$(sort -u "${PBS_NODEFILE}" | wc -l)"
    else
        num_nodes=1
    fi
fi

RDZV_HOST="${RDZV_HOST:-$(hostname -i)}"
RDZV_PORT="${RDZV_PORT:-3000}"

CONFIGS=(
    "${MODEL_ROOT}/online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b/config.yaml"
)

echo "Running focused DnR hierarchical comparison"
echo "  dry run: ${DRY_RUN}"
echo "  continue on error: ${CONTINUE_ON_ERROR}"
echo "  log root: ${LOG_ROOT}"

for config in "${CONFIGS[@]}"; do
    workdir="$(dirname "${config}")"
    rdzv_id="$(basename "${workdir}")"
    logfile="${LOG_ROOT}/${rdzv_id}.$(date +%Y%m%d-%H%M%S).log"
    cmd=(
        torchrun
        --nproc_per_node="${ngpu}"
        --nnodes="${num_nodes}"
        --rdzv_id "${rdzv_id}"
        --rdzv_backend=c10d
        --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}"
        "${TRAIN_ENTRY}" "${config}"
    )

    printf 'Command: '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [ "${DRY_RUN}" -eq 1 ]; then
        continue
    fi

    if "${cmd[@]}" 2>&1 | tee "${logfile}"; then
        :
    elif [ "${CONTINUE_ON_ERROR}" -eq 0 ]; then
        exit 1
    fi
done
