#!/bin/bash

if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
fi

if [ -z "$wd" ]; then
    wd=$(dirname "${0}")
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L | wc -l)
else
    ngpu=1
fi
if [ "$ngpu" -lt 1 ]; then
    ngpu=1
fi
if [ -n "$PBS_NODEFILE" ] && [ -f "$PBS_NODEFILE" ]; then
    num_nodes=$(sort -u "$PBS_NODEFILE" | wc -l)
else
    num_nodes=1
fi
RDZV_ID=$(basename "$wd")

torchrun --nproc_per_node="$ngpu" --nnodes="$num_nodes" \
    --rdzv_id "$RDZV_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname -i):3000 \
    aiaccel/aiaccel/torch/apps/train.py "$wd/config.yaml"
