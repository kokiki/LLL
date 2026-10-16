#!/usr/bin/env bash
# v10 방향 일반화 개선 시험: (A) 빔 정렬 입력 BSC-UNet-A, (DA) BSC-UNet + 방향 증강, 각각 k=1, 16 (개선 학습 설정)
#   -> test·guard 평가 (baseline 7종 + 기존 BSC-UNet-I, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 16; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --aligned --out_dir ./runs/v10_Tp20_bscAI_k${K}ms > ./runs/train_v10_Tp20_bscAI_k${K}ms.log 2>&1 &
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --dir_aug --out_dir ./runs/v10_Tp20_bscDA_k${K}ms > ./runs/train_v10_Tp20_bscDA_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
CK=""; for K in 1 16; do for M in bscI bscAI bscDA; do CK="$CK --ckpt ${M}_k${K}=./runs/v10_Tp20_${M}_k${K}ms/best_G.pt:${K}"; done; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $CK --leak_check --out ./runs/eval_v10_fix_${S}.json > ./runs/eval_v10_fix_${S}.log 2>&1
done
echo "[$(date)] eval done"
