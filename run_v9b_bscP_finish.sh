#!/usr/bin/env bash
# 최종 모델 BSC-UNet-P (κ = 0.155, v9b): k = 12 학습 (k = 1, 16 완료, k = 2, 4, 8 학습 중; k = 24, 32는 제외)
#   -> k = 2, 4, 8, 12 학습이 모두 끝나면 6 horizon (k = 1–16) test·guard 평가 (baseline 8종 + BSC-UNet-I, 누수 점검) -> 방위각 시험
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
python -u train_beam.py --data_root $D --horizon_ms 12 --Tp 20 $IMP --poly_shortcut \
  --out_dir ./runs/v9b_Tp20_bscP_k12ms > ./runs/train_v9b_Tp20_bscP_k12ms.log 2>&1
n_run() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'train_beam.py.*bscP_k' }).Count" | tr -d '\r'; }
while [ "$(n_run)" != "0" ]; do sleep 120; done
echo "[$(date)] training done"
bscI() { case $1 in 1|2|4) echo "./runs/v9b_Tp20_bscI_lrfix_k$1ms/best_G.pt";; *) echo "./runs/v9b_Tp20_bscI_k$1ms/best_G.pt";; esac; }
ck() { for K in "$@"; do echo -n " --ckpt bscP_k${K}=./runs/v9b_Tp20_bscP_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=$(bscI $K):${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 1 2 4 8 12 16) --leak_check --out ./runs/eval_v9b_final_short_${S}.json > ./runs/eval_v9b_final_short_${S}.log 2>&1
done
echo "[$(date)] eval done"
python -u eval_bearing_final.py > ./runs/eval_bearing_v9b_final.log 2>&1
echo "[$(date)] bearing done"
