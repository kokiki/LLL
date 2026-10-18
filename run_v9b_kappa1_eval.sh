#!/usr/bin/env bash
# κ = 1 주 결과 평가 (BSC-UNet-P, BSC-UNet-I; k = 1–16; test·guard; baseline 9종은 κ = 1 데이터로 적합) -> 방위각 시험
#   -> 그 다음 "딥러닝만"(지름길 없는 UNet) κ = 1 학습 6개 (4개씩) -> 딥러닝만 평가
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
ck() { for K in "$@"; do echo -n " --ckpt bscP_k${K}=./runs/v9b_Tp20_bscP_kappa1_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=./runs/v9b_Tp20_bscI_kappa1_k${K}ms/best_G.pt:${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --doppler_rate_kappa 1 --leak_check $(ck 1 2 4 8 12 16) \
    --out ./runs/eval_v9b_kappa1_main_${S}.json > ./runs/eval_v9b_kappa1_main_${S}.log 2>&1
done
echo "[$(date)] eval done"
python -u eval_bearing_final.py --doppler_rate_kappa 1 > ./runs/eval_bearing_v9b_kappa1.log 2>&1
echo "[$(date)] bearing done"
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
export D IMP
printf '%s\n' 1 2 4 8 12 16 | xargs -P 4 -I{} bash -c 'python -u train_beam.py --data_root $D --horizon_ms {} --Tp 20 $IMP --doppler_rate_kappa 1 --no_shortcut --out_dir ./runs/v9b_Tp20_unet_kappa1_k{}ms > ./runs/train_v9b_Tp20_unet_kappa1_k{}ms.log 2>&1'
echo "[$(date)] UNet-only training done"
cku() { for K in "$@"; do echo -n " --ckpt unet_k${K}=./runs/v9b_Tp20_unet_kappa1_k${K}ms/best_G.pt:${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --doppler_rate_kappa 1 --leak_check $(cku 1 2 4 8 12 16) \
    --out ./runs/eval_v9b_kappa1_unet_${S}.json > ./runs/eval_v9b_kappa1_unet_${S}.log 2>&1
done
echo "[$(date)] UNet-only eval done"
