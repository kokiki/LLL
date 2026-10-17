#!/usr/bin/env bash
# ablation "딥러닝만": 지름길 없는 UNet (csi_models.GeneratorUNet, train_beam.py --no_shortcut), v9b κ = 0.155, k = 1, 16
#   BSC-UNet-P 나머지 horizon 학습(run_v9b_bscP_all.sh)이 끝난 뒤 시작 (GPU 메모리). 학습 설정은 제안 모델과 동일
#   -> test·guard 평가 (baseline 8종, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
n_run() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'train_beam.py.*bscP_k' }).Count" | tr -d '\r'; }
while [ "$(n_run)" != "0" ]; do sleep 120; done
echo "[$(date)] start UNet-only training"
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 16; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --no_shortcut \
    --out_dir ./runs/v9b_Tp20_unet_k${K}ms > ./runs/train_v9b_Tp20_unet_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --leak_check \
    --ckpt unet_k1=./runs/v9b_Tp20_unet_k1ms/best_G.pt:1 --ckpt unet_k16=./runs/v9b_Tp20_unet_k16ms/best_G.pt:16 \
    --out ./runs/eval_v9b_unet_${S}.json > ./runs/eval_v9b_unet_${S}.log 2>&1
done
echo "[$(date)] eval done"
