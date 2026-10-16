#!/usr/bin/env bash
# v10 BSC-UNet-AIR: BSC-UNet-AI (빔 정렬) + 배열 비이상성 증강(--array_aug), k=1, 16
#   -> eval_array_impair_v10.py (이상적 / 위상·이득 오차 / 고장 소자, test·guard)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 16; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --aligned --array_aug --out_dir ./runs/v10_Tp20_bscAIR_k${K}ms > ./runs/train_v10_Tp20_bscAIR_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
