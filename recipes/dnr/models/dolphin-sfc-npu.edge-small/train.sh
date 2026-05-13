#!/bin/bash

if [ -n "$PBS_O_WORKDIR" ]; then
    cd $PBS_O_WORKDIR
fi

if [ -z "$wd" ]; then
    wd=`dirname "${0}"`
fi

ngpu=$(nvidia-smi -L | wc -l)
num_nodes=$(sort -u $PBS_NODEFILE | wc -l)
RDZV_ID=$(basename "$wd")

torchrun --nproc_per_node=$ngpu --nnodes=$num_nodes \
    --rdzv_id $RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname -i):3000 \
    aiaccel/aiaccel/torch/apps/train.py $wd/config.yaml
