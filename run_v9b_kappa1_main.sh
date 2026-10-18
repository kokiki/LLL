#!/usr/bin/env bash
# 주 결과를 κ = 1(보상 없는 실제 Doppler 변화율)로 전환: v9b 데이터에 로더에서 2차 위상 추가 (--doppler_rate_kappa 1)
#   학습 14개 (4개씩 동시): BSC-UNet-P k=2,4,8,12 -> BSC-UNet-I k=2,4,8,12 -> 딥러닝만(UNet) k=1,2,4,8,12,16
#   (BSC-UNet-P·I k=1,16은 run_v9b_bscP.sh / run_v9b_bscI_kappa1.sh에서 완료)
#   "딥러닝만" κ = 0.155 학습(run_v9b_unet_only_rest.sh)이 끝난 뒤 시작 -> κ = 1 test·guard 평가 (k=1–16) -> 방위각 시험
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
n_run() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'train_beam.py.*unet_k[0-9]' }).Count" | tr -d '\r'; }
while [ "$(n_run)" != "0" ]; do sleep 120; done
echo "[$(date)] start kappa=1 trainings"
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
JOBS=$(mktemp)
for K in 2 4 8 12; do echo "--poly_shortcut|bscP_kappa1_k${K}ms|$K"; done >> $JOBS
for K in 2 4 8 12; do echo "|bscI_kappa1_k${K}ms|$K"; done >> $JOBS
for K in 1 2 4 8 12 16; do echo "--no_shortcut|unet_kappa1_k${K}ms|$K"; done >> $JOBS
export D IMP
cat $JOBS | xargs -P 4 -I{} bash -c 'IFS="|" read -r F N K <<< "{}"; python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --doppler_rate_kappa 1 $F --out_dir ./runs/v9b_Tp20_$N > ./runs/train_v9b_Tp20_$N.log 2>&1'
rm -f $JOBS
echo "[$(date)] training done"
ck() { for K in "$@"; do echo -n " --ckpt bscP_k${K}=./runs/v9b_Tp20_bscP_kappa1_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=./runs/v9b_Tp20_bscI_kappa1_k${K}ms/best_G.pt:${K} --ckpt unet_k${K}=./runs/v9b_Tp20_unet_kappa1_k${K}ms/best_G.pt:${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 --doppler_rate_kappa 1 --leak_check $(ck 1 2 4 8 12 16) \
    --out ./runs/eval_v9b_kappa1_main_${S}.json > ./runs/eval_v9b_kappa1_main_${S}.log 2>&1
done
echo "[$(date)] eval done"
python -u eval_bearing_final.py --doppler_rate_kappa 1 > ./runs/eval_bearing_v9b_kappa1.log 2>&1
echo "[$(date)] bearing done"
