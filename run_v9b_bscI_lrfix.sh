#!/usr/bin/env bash
# v9b BSC-UNet-I 재학습: 코사인 학습률 재상승 수정 (cosine 80 epoch 후 1e-6 유지, train_beam.py LambdaLR)
#   기존 최종 가중치가 81 epoch 이후에 저장된 k = 1, 2, 4만 재학습 (k >= 8은 81 epoch 전에 저장되어 새 스케줄과 동일 경로)
#   -> test·guard 평가 (baseline 8종, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 2 4; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --out_dir ./runs/v9b_Tp20_bscI_lrfix_k${K}ms > ./runs/train_v9b_Tp20_bscI_lrfix_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
CK=""; for K in 1 2 4; do CK="$CK --ckpt bscIlrfix_k${K}=./runs/v9b_Tp20_bscI_lrfix_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=./runs/v9b_Tp20_bscI_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $CK --leak_check --out ./runs/eval_v9b_lrfix_${S}.json > ./runs/eval_v9b_lrfix_${S}.log 2>&1
done
echo "[$(date)] eval done"
