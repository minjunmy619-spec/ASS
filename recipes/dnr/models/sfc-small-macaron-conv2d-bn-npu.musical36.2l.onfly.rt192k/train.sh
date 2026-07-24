#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RECIPE_DIR/../../../.." && pwd)"

cd "$REPO_ROOT"
exec .venv/bin/python -m aiaccel.torch.apps.train "$RECIPE_DIR/config.yaml" "$@"
