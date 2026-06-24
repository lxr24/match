# 点云降噪竞赛 — StraightPCFCore (Jittor)

基于 StraightPCF 四阶段课程（VM → CVM → Core → Joint），对接 `dataset_clean` / `test_noisy` 与 `datalist` 三分区。

## 数据目录

```text
dataset_clean/shapenet/<synset>/<model_id>/models/model_normalized.obj
test_noisy/shapenet/<synset>/<model_id>/noisy.npy
datalist/train/      # 15733 行
datalist/validation/ # 100 行
datalist/test/       # 200 行
```

datalist 每行格式：`shapenet/<synset_id>/<model_id>`

## 环境

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
pip install -r requirements.txt
# Ubuntu: sudo apt install python3-dev
export PYTHONPATH=/workspace
```

## 快速开始

### 1. 审计 datalist 并生成 5k/10k 子集

```bash
python scripts/audit_datalist.py --datalist_dir ./datalist
```

生成 `datalist/train_strat5k_seed123.txt` 与 `datalist/train_strat10k_seed123.txt`。

### 2. 构建 validation 评测缓存（100 条 × 50k 点）

```bash
python scripts/build_val_cache.py \
  --dataset_root ./dataset_clean \
  --datalist_dir ./datalist \
  --val_cache_root ./val_cache
```

### 3. 四阶段训练（5k 基线 ~40min）

```bash
bash scripts/train_all.sh
# 或逐步：
python run.py --task configs/train_vm.yaml --seed 123
python run.py --task configs/train_cvm.yaml --seed 123
python run.py --task configs/train_core.yaml --seed 123
python run.py --task configs/train_joint.yaml --seed 123
```

推荐 checkpoint：`experiments/straightpcf_joint_short5k_s123/checkpoint_1.pkl`

### 4. 扩展 10k + low_lr 冲分

```bash
python run.py --task configs/train_vm_10k.yaml --seed 123
python run.py --task configs/train_cvm_10k.yaml --seed 123
python run.py --task configs/train_core_10k.yaml --seed 123
python run.py --task configs/train_joint_10k.yaml --seed 123
python run.py --task configs/train_joint_lowlr.yaml --seed 123
```

### 5. 本地验分（validation@100）

```bash
python run.py --task configs/validate.yaml \
  --checkpoint experiments/straightpcf_joint_short5k_s123/checkpoint_1.pkl
```

### 6. 测试集推理与提交（test@200）

```bash
python scripts/predict_test.py \
  --task configs/predict.yaml \
  --checkpoint experiments/straightpcf_joint_short5k_s123/checkpoint_1.pkl \
  --test_root ./test_noisy \
  --output_root ./results

bash scripts/submit.sh ./results ./result.zip
```

## 关键实现细节

- 训练噪声：VM/CVM/Core 用 Laplace σ∈[0.005,0.020]；Joint 用 65/35 mixture
- 训练输入：`pc_mix = t·clean + (1-t)·noisy`，patch 局部中心化
- CVM 从 VM 复制权重；推理用 `best` 聚合，`inner_steps=2`
- 8GB 显存：`batch_size=4`，OOM 时改为 2

## 冒烟测试（无真实数据）

```bash
PYTHONPATH=/workspace python scripts/smoke_test.py
```
