#!/usr/bin/env bash
# BSC-UNet + 개선 학습 설정 (BSC-UNet-I: cosine lr 1e-4->1e-6/80ep, EMA 0.999, beta1 0.9, grad clip 1.0, patience 20)
#   k=16(진행 중) 이후 k=4, 12 / 동시에 k=1, 2, 8 -> 6 horizon test+guard 평가, 기존 BSC-UNet과 paired, 안테나 오차 지도, 일반화 그림
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
tr() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp $TP $IMP --out_dir ./runs/v9b_Tp${TP}_bscI_k${K}ms > ./runs/train_v9b_Tp${TP}_bscI_k${K}ms.log 2>&1
done; }
( until [ -f ./runs/v9b_Tp${TP}_bscI_k16ms/result.json ]; do sleep 60; done; tr 4 12 ) &
tr 1 2 8 &
wait
until [ -f ./runs/v9b_Tp${TP}_bscI_k16ms/result.json ]; do sleep 60; done
echo "[$(date)] training done"
CK=""; for K in 1 2 4 8 12 16; do CK="$CK --ckpt BSCI_k${K}=./runs/v9b_Tp${TP}_bscI_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_bscI_${S}.json > ./runs/eval_v9b_Tp${TP}_bscI_${S}.log 2>&1
  python -u compare_models.py --data_root $D --split $S --pair "BSCI=./runs/v9b_Tp${TP}_bscI_k{K}ms/best_G.pt" \
    --pair "BSC=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/compare_bscI_bsc_${S}.json > ./runs/compare_bscI_bsc_${S}.log 2>&1
done
python -u plot_antenna_error.py --ks 4 16 --out figures/antenna_error_map_v9b_bscI.png > ./runs/antenna_error_bscI.log 2>&1
python -u plot_generalization.py --data_root $D --ckpt "runs/v9b_Tp${TP}_bscI_k{K}ms/best_G.pt" --out figures/generalization_nmse_vs_elevation_v9b_bscI.png > ./runs/plot_generalization_bscI.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v9b_Tp${TP}_bscI_k*ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_Tp${TP}_bscI_*.log ./runs/compare_bscI_bsc_*.log ./runs/antenna_error_bscI.log ./runs/plot_generalization_bscI.log
echo done
