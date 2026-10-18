#!/usr/bin/env bash
# ablation "딥러닝만" (지름길 없는 UNet, --no_shortcut) 나머지 horizon k = 2, 4, 8, 12 (k = 1, 16은 run_v9b_unet_only.sh에서 완료)
#   학습 설정은 제안 모델과 동일 -> k = 1–16 전체 test·guard 평가 (baseline 9종, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 2 4 8 12; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --no_shortcut \
    --out_dir ./runs/v9b_Tp20_unet_k${K}ms > ./runs/train_v9b_Tp20_unet_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
ck() { for K in "$@"; do echo -n " --ckpt unet_k${K}=./runs/v9b_Tp20_unet_k${K}ms/best_G.pt:${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --leak_check $(ck 1 2 4 8 12 16) \
    --out ./runs/eval_v9b_unet_${S}.json > ./runs/eval_v9b_unet_${S}.log 2>&1
done
echo "[$(date)] eval done"
