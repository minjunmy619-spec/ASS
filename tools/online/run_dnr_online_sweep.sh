#!/usr/bin/env bash

set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_ROOT="recipes/dnr/models"
TRAIN_ENTRY="aiaccel/aiaccel/torch/apps/train.py"
LOG_ROOT_DEFAULT="logs/dnr_online_sweep"

FAMILY_FILTER="all"
DRY_RUN=0
CONTINUE_ON_ERROR=1
LOG_ROOT="${LOG_ROOT_DEFAULT}"

usage() {
    cat <<EOF
Usage: $0 [family] [options]

family:
  all | plain | soft | soft-query | soft-dilated | soft-gru | hierarchical-soft | hierarchical-soft-ffi | hierarchical-soft-parallel-ffi | hard

options:
  --dry-run              Print the selected training commands without launching
  --stop-on-error        Stop the sweep on the first failed config
  --log-root <dir>       Directory for per-run log files
  -h, --help             Show this help
EOF
}

POSITIONAL=()
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
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL[@]}" -ge 1 ]; then
    FAMILY_FILTER="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -gt 1 ]; then
    echo "Too many positional arguments" >&2
    usage >&2
    exit 1
fi

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

case "${FAMILY_FILTER}" in
    all|plain|soft|soft-query|soft-dilated|soft-gru|hierarchical-soft|hierarchical-soft-ffi|hierarchical-soft-parallel-ffi|hard)
        ;;
    *)
        echo "Unsupported family filter: ${FAMILY_FILTER}" >&2
        echo "Usage: $0 [all|plain|soft|soft-query|soft-dilated|soft-gru|hierarchical-soft|hierarchical-soft-ffi|hierarchical-soft-parallel-ffi|hard]" >&2
        exit 1
        ;;
esac

declare -a CONFIGS=(
    "${MODEL_ROOT}/online-sfc2d.causal96dim.12l.musical64/config.yaml"
    "${MODEL_ROOT}/online-soft-band-sfc2d.causal96dim.12l.musical64/config.yaml"
    "${MODEL_ROOT}/online-soft-band-sfc2d.mel.causal96dim.12l/config.yaml"
    "${MODEL_ROOT}/online-soft-band-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml"
    "${MODEL_ROOT}/online-soft-band-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml"
    "${MODEL_ROOT}/online-soft-band-query-sfc2d.causal96dim.12l.musical64/config.yaml"
    "${MODEL_ROOT}/online-soft-band-query-sfc2d.mel.causal96dim.12l/config.yaml"
    "${MODEL_ROOT}/online-soft-band-query-sfc2d.rt192k.causal24dim.6l.64b/config.yaml"
    "${MODEL_ROOT}/online-soft-band-query-sfc2d.rt128k.causal16dim.6l.64b/config.yaml"
    "${MODEL_ROOT}/online-soft-band-query-sfc2d.rt192k.fp512keep475.causal24dim.6l.64b/config.yaml"
    "${MODEL_ROOT}/online-soft-band-query-sfc2d.rt128k.fp512keep475.causal16dim.6l.64b/config.yaml"
    "${MODEL_ROOT}/online-soft-band-dilated-sfc2d.causal96dim.12l.musical64/config.yaml"
    "${MODEL_ROOT}/online-soft-band-dilated-sfc2d.mel.causal96dim.12l/config.yaml"
    "${MODEL_ROOT}/online-soft-band-gru-sfc2d.causal96dim.12l.musical64/config.yaml"
    "${MODEL_ROOT}/online-soft-band-gru-sfc2d.mel.causal96dim.12l/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-sfc2d.causal96dim.1-2-2l.musical128/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-sfc2d.mel.causal96dim.1-2-2l/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-ffi-sfc2d.speech-lowfreq-narrow.causal96dim.1-2-2l/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt192k.speech-lowfreq-narrow.causal20dim.0-1-1l.128-96-48b/config.yaml"
    "${MODEL_ROOT}/online-hierarchical-soft-band-parallel-ffi-sfc2d.rt128k.speech-lowfreq-narrow.causal14dim.0-1-1l.128-96-48b/config.yaml"
    "${MODEL_ROOT}/online-hard-band-sfc2d.causal96dim.12l.musical64/config.yaml"
    "${MODEL_ROOT}/online-hard-band-sfc2d.mel.causal96dim.12l/config.yaml"
)

matches_family() {
    local path="$1"
    case "${FAMILY_FILTER}" in
        all)
            return 0
            ;;
        plain)
            [[ "${path}" == *"/online-sfc2d."* ]]
            ;;
        soft)
            [[ "${path}" == *"/online-soft-band-sfc2d."* ]]
            ;;
        soft-query)
            [[ "${path}" == *"/online-soft-band-query-sfc2d."* ]]
            ;;
        soft-dilated)
            [[ "${path}" == *"/online-soft-band-dilated-sfc2d."* ]]
            ;;
        soft-gru)
            [[ "${path}" == *"/online-soft-band-gru-sfc2d."* ]]
            ;;
        hierarchical-soft)
            [[ "${path}" == *"/online-hierarchical-soft-band-sfc2d."* ]]
            ;;
        hierarchical-soft-ffi)
            [[ "${path}" == *"/online-hierarchical-soft-band-ffi-sfc2d."* ]]
            ;;
        hierarchical-soft-parallel-ffi)
            [[ "${path}" == *"/online-hierarchical-soft-band-parallel-ffi-sfc2d."* ]]
            ;;
        hard)
            [[ "${path}" == *"/online-hard-band-sfc2d."* ]]
            ;;
    esac
}

declare -a SELECTED=()
for config in "${CONFIGS[@]}"; do
    if matches_family "${config}"; then
        SELECTED+=("${config}")
    fi
done

if [ "${#SELECTED[@]}" -eq 0 ]; then
    echo "No configs matched family='${FAMILY_FILTER}'" >&2
    exit 1
fi

echo "Running DnR online sweep with ${#SELECTED[@]} config(s)"
echo "  family filter: ${FAMILY_FILTER}"
echo "  dry run: ${DRY_RUN}"
echo "  continue on error: ${CONTINUE_ON_ERROR}"
echo "  log root: ${LOG_ROOT}"
echo "  ngpu: ${ngpu}"
echo "  num_nodes: ${num_nodes}"
echo "  rdzv: ${RDZV_HOST}:${RDZV_PORT}"

mkdir -p "${LOG_ROOT}"

declare -a FAILED=()
declare -a SUCCEEDED=()

for config in "${SELECTED[@]}"; do
    if [ ! -f "${config}" ]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi

    workdir="$(dirname "${config}")"
    rdzv_id="$(basename "${workdir}")"

    echo
    echo "============================================================"
    echo "Launching: ${config}"
    echo "Workdir:   ${workdir}"
    echo "RDZV ID:   ${rdzv_id}"
    echo "============================================================"

    timestamp="$(date +%Y%m%d-%H%M%S)"
    logfile="${LOG_ROOT}/${rdzv_id}.${timestamp}.log"

    cmd=(
        torchrun
        --nproc_per_node="${ngpu}"
        --nnodes="${num_nodes}"
        --rdzv_id "${rdzv_id}"
        --rdzv_backend=c10d
        --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}"
        "${TRAIN_ENTRY}" "${config}"
    )

    echo "Log file:  ${logfile}"
    printf 'Command:   '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [ "${DRY_RUN}" -eq 1 ]; then
        continue
    fi

    if "${cmd[@]}" 2>&1 | tee "${logfile}"; then
        SUCCEEDED+=("${config}")
    else
        FAILED+=("${config}")
        echo "Training failed for ${config}" >&2
        if [ "${CONTINUE_ON_ERROR}" -eq 0 ]; then
            break
        fi
    fi
done

if [ "${DRY_RUN}" -eq 1 ]; then
    exit 0
fi

echo
echo "Sweep summary"
echo "  succeeded: ${#SUCCEEDED[@]}"
echo "  failed:    ${#FAILED[@]}"

if [ "${#FAILED[@]}" -gt 0 ]; then
    printf 'Failed configs:\n'
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
