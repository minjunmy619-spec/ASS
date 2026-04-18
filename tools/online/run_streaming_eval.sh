#! /bin/bash

# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <musdb|dnr> <model_path> [chunk_frames] [device]" >&2
    exit 1
fi

dataset=$1
model_path=$2
chunk_frames=${3:-8}
device=${4:-cuda}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

case "$dataset" in
    musdb|musdb18hq)
        split=test
        dataset_path="${repo_root}/recipes/musdb18hq/data/${split}"
        pattern="mixture.wav"
        evaluator_module="recipes.musdb18hq.scripts.evaluate_sdr"
        metrics=("usdr" "csdr" "sisdr")
        stem_args=()
        eval_extra_args=(--verbose)
        ;;
    dnr)
        split=tt
        dataset_path="${repo_root}/recipes/dnr/data/dnr_v2/${split}"
        pattern="mix.wav"
        evaluator_module="recipes.dnr.scripts.evaluate_sdr"
        metrics=("usdr" "sisdr")
        stem_args=(--stem-name speech --stem-name music --stem-name sfx)
        eval_extra_args=()
        ;;
    *)
        echo "Unsupported dataset: ${dataset}. Use musdb or dnr." >&2
        exit 2
        ;;
esac

if [[ "$model_path" =~ \.(ckpt|pth|pt)$ ]]; then
    output_base="$(dirname "$(dirname "$model_path")")"
elif [[ "$(basename "$model_path")" =~ ^(config|merged_config)\.yaml$ ]]; then
    output_base="$(dirname "$model_path")"
else
    output_base="$model_path"
fi

wav_output_path="${output_base}/wav_stream_cf${chunk_frames}/${split}"
manifest_path="${wav_output_path}/run_manifest.json"

mkdir -p "${wav_output_path}"

python "${repo_root}/tools/online/run_streaming_inference.py" \
    "${model_path}" \
    "${dataset_path}" \
    "${wav_output_path}" \
    --device "${device}" \
    --pattern "${pattern}" \
    --chunk-frames "${chunk_frames}" \
    --output-group "${split}" \
    --manifest-out "${manifest_path}" \
    --overwrite \
    "${stem_args[@]}"

for metric in "${metrics[@]}"; do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m "${evaluator_module}" \
        --est_dir "${wav_output_path}" \
        --ref_dir "${dataset_path}" \
        --metric "${metric}" \
        "${eval_extra_args[@]}"
done
