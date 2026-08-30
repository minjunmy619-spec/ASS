#!/bin/bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$recipe_dir/../../../.." && pwd)

cd "$project_root"
export PYTHONPATH="$project_root/aiaccel${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m aiaccel.torch.apps.train "$recipe_dir/config.yaml" "$@"
