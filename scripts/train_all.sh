#!/usr/bin/env bash
set -euo pipefail
SEED=${SEED:-123}
PYTHON=${PYTHON:-python3}
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"

$PYTHON run.py --task configs/train_vm.yaml --seed $SEED
$PYTHON run.py --task configs/train_cvm.yaml --seed $SEED
$PYTHON run.py --task configs/train_core.yaml --seed $SEED
$PYTHON run.py --task configs/train_joint.yaml --seed $SEED
