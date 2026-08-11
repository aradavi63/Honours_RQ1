#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="rq1-fedshe-ckks"
PYFHEL_ROOT="${PYFHEL_ROOT:-$HOME/src/Pyfhel-3.4.2}"
PYFHEL_COMMIT="a0ce75f36c081c30ae1cbfbcf1926f2698c94420"

if [[ ! -x "$CONDA_BIN" ]]; then
    echo "Conda was not found at $CONDA_BIN" >&2
    exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA_BIN" env update --name "$ENV_NAME" --file "$ROOT/environments/fedshe-ckks-linux.yml" --prune
else
    "$CONDA_BIN" env create --file "$ROOT/environments/fedshe-ckks-linux.yml"
fi

if [[ ! -d "$PYFHEL_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$PYFHEL_ROOT")"
    git clone --recursive --branch v3.4.2 --depth 1 \
        https://github.com/ibarrond/Pyfhel.git "$PYFHEL_ROOT"
fi

git -C "$PYFHEL_ROOT" submodule update --init --recursive
actual_commit="$(git -C "$PYFHEL_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$PYFHEL_COMMIT" ]]; then
    echo "Expected Pyfhel $PYFHEL_COMMIT, found $actual_commit" >&2
    exit 1
fi

"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install \
    --no-build-isolation "$PYFHEL_ROOT"
"$CONDA_BIN" run --name "$ENV_NAME" python -c \
    "from Pyfhel import Pyfhel; print('Pyfhel CKKS preflight passed')"

echo "Environment $ENV_NAME is ready"
