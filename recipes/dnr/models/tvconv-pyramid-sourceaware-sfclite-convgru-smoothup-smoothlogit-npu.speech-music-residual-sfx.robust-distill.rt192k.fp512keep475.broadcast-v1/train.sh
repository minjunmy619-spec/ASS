#!/bin/bash

if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR" || exit 1
fi

if [ -z "$wd" ]; then
    wd=$(dirname "${0}")
fi

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ngpu" -eq 0 ]; then
    ngpu=1
fi
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "$PBS_NODEFILE" ]; then
    num_nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
else
    num_nodes=1
fi
RDZV_ID=$(basename "$wd")
rdzv_host=$(hostname -i | awk '{print $1}')

torchrun --nproc_per_node="$ngpu" --nnodes="$num_nodes" \
    --rdzv_id "$RDZV_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR:-$rdzv_host}:${MASTER_PORT:-3000}" \
    aiaccel/aiaccel/torch/apps/train.py "$wd/config.yaml"
