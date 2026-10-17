#!/usr/bin/env bash
# v9b BSC-UNet-I, Doppler 변화율 실제 크기(κ = 1) 조건 재학습: k = 1, 16 (로더에서 2차 위상 추가, data.rescale_doppler_rate)
#   학습률 수정 재학습(run_v9b_bscI_lrfix.sh)의 학습이 끝난 뒤 시작 (GPU 메모리)
#   -> κ = 1 test·guard 평가: baseline 8종은 κ = 1 train/val로 재적합, κ = 0.155로 학습한 모델도 참고로 포함
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
n_lrfix() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'train_beam.py.*bscI_lrfix' }).Count" | tr -d ''; }
while [ "$(n_lrfix)" != "0" ]; do sleep 120; done
echo "[$(date)] start kappa=1 training"
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 16; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --doppler_rate_kappa 1 \
    --out_dir ./runs/v9b_Tp20_bscI_kappa1_k${K}ms > ./runs/train_v9b_Tp20_bscI_kappa1_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
for S in test guard; do
  CK="--ckpt bscIkappa1_k1=./runs/v9b_Tp20_bscI_kappa1_k1ms/best_G.pt:1 --ckpt bscIkappa1_k16=./runs/v9b_Tp20_bscI_kappa1_k16ms/best_G.pt:16"
  CK="$CK --ckpt bscI_k1=./runs/v9b_Tp20_bscI_lrfix_k1ms/best_G.pt:1 --ckpt bscI_k16=./runs/v9b_Tp20_bscI_k16ms/best_G.pt:16"
  python -u evaluate.py --data_root $D --split $S --Tp 20 $CK --doppler_rate_kappa 1 --leak_check \
    --out ./runs/eval_v9b_kappa1_${S}.json > ./runs/eval_v9b_kappa1_${S}.log 2>&1
done
echo "[$(date)] eval done"
