#!/usr/bin/env bash
# v10(구면 궤도 기하, 방위각·off-nadir 다양) κ = 1에서 BSC-UNet-AP(빔 정렬 + 2차 외삽 지름길 + UNet) 학습·검증, k = 1–16
#   딥러닝 baseline(run_v9b_dl_baselines.sh)이 끝난 뒤 시작. 학습 6개 (4개씩) -> v10 test·guard 평가
#   (baseline 9종은 v10 κ = 1 train/val로 적합, 비교용으로 v9b로 학습한 BSC-UNet-P도 포함)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
n_prev() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -match 'run_v9b_dl_baselines\.sh' }).Count" | tr -d '\r'; }
while [ "$(n_prev)" != "0" ]; do sleep 120; done
echo "[$(date)] start v10 BSC-UNet-AP training"
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
export D IMP
printf '%s\n' 1 2 4 8 12 16 | xargs -P 4 -I{} bash -c 'python -u train_beam.py --data_root $D --horizon_ms {} --Tp 20 $IMP --doppler_rate_kappa 1 --aligned --poly_shortcut --out_dir ./runs/v10_Tp20_bscAP_kappa1_k{}ms > ./runs/train_v10_Tp20_bscAP_kappa1_k{}ms.log 2>&1'
echo "[$(date)] training done"
CK=""; for K in 1 2 4 8 12 16; do CK="$CK --ckpt bscAP_k${K}=./runs/v10_Tp20_bscAP_kappa1_k${K}ms/best_G.pt:${K} --ckpt bscPv9b_k${K}=./runs/v9b_Tp20_bscP_kappa1_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --doppler_rate_kappa 1 --leak_check $CK \
    --out ./runs/eval_v10_bscAP_kappa1_${S}.json > ./runs/eval_v10_bscAP_kappa1_${S}.log 2>&1
done
echo "[$(date)] eval done"
