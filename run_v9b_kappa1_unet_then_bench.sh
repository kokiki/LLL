#!/usr/bin/env bash
# "딥러닝만"(지름길 없는 UNet) κ = 1 학습 6개 (4개씩) -> 평가 -> GPU가 빈 상태에서 BSC-UNet-P 추론 시간 측정
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
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
python -u bench_inference_p.py > ./runs/bench_inference_p.log 2>&1
echo "[$(date)] bench done"
