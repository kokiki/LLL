#!/usr/bin/env bash
# v9b BSC-UNet-P (빔 지름길 + 2차 위상 외삽, csi_models.BeamShortcutPolyUNet): k = 1, 16 x Doppler 변화율 κ = 0.155 / 1
#   -> 조건별 test·guard 평가 (baseline 8종은 같은 κ 데이터로 적합; 같은 κ로 학습한 BSC-UNet-I와 비교)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 16; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --poly_shortcut \
    --out_dir ./runs/v9b_Tp20_bscP_k${K}ms > ./runs/train_v9b_Tp20_bscP_k${K}ms.log 2>&1 &
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --poly_shortcut --doppler_rate_kappa 1 \
    --out_dir ./runs/v9b_Tp20_bscP_kappa1_k${K}ms > ./runs/train_v9b_Tp20_bscP_kappa1_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --leak_check \
    --ckpt bscP_k1=./runs/v9b_Tp20_bscP_k1ms/best_G.pt:1 --ckpt bscP_k16=./runs/v9b_Tp20_bscP_k16ms/best_G.pt:16 \
    --ckpt bscI_k1=./runs/v9b_Tp20_bscI_lrfix_k1ms/best_G.pt:1 --ckpt bscI_k16=./runs/v9b_Tp20_bscI_k16ms/best_G.pt:16 \
    --out ./runs/eval_v9b_bscP_${S}.json > ./runs/eval_v9b_bscP_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 --leak_check --doppler_rate_kappa 1 \
    --ckpt bscP_k1=./runs/v9b_Tp20_bscP_kappa1_k1ms/best_G.pt:1 --ckpt bscP_k16=./runs/v9b_Tp20_bscP_kappa1_k16ms/best_G.pt:16 \
    --ckpt bscI_k1=./runs/v9b_Tp20_bscI_kappa1_k1ms/best_G.pt:1 --ckpt bscI_k16=./runs/v9b_Tp20_bscI_kappa1_k16ms/best_G.pt:16 \
    --out ./runs/eval_v9b_bscP_kappa1_${S}.json > ./runs/eval_v9b_bscP_kappa1_${S}.log 2>&1
done
echo "[$(date)] eval done"
