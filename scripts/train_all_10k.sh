#!/usr/bin/env bash
# 10k four-stage training
set -euo pipefail
SEED=${SEED:-123}
PYTHON=${PYTHON:-python3}
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

$PYTHON run.py --task configs/train_vm_10k.yaml --seed $SEED
$PYTHON run.py --task configs/train_cvm_10k.yaml --seed $SEED
$PYTHON run.py --task configs/train_core_10k.yaml --seed $SEED
$PYTHON run.py --task configs/train_joint_10k.yaml --seed $SEED
