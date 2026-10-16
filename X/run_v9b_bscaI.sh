#!/usr/bin/env bash
# BSC-UNet-A 개선 학습 설정 (cosine lr 1e-4->1e-6/80ep, EMA 0.999, beta1 0.9, grad clip 1.0, patience 20): k=1,2 동시 -> k=16
#   -> 기존 BSC-UNet과 test+guard 비교 (paired), 안테나별 오차 지도
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
tr_imp() { python -u train_beam.py --data_root $D --horizon_ms $1 --Tp $TP --aligned $IMP --out_dir ./runs/v9b_Tp${TP}_bscaI_k$1ms > ./runs/train_v9b_Tp${TP}_bscaI_k$1ms.log 2>&1; }
tr_imp 1 &
P1=$!
tr_imp 2 &
P2=$!
wait -n $P1 $P2
tr_imp 16
wait
echo "[$(date)] training done"
CK=""; for K in 1 2 16; do CK="$CK --ckpt bscaI_k${K}=./runs/v9b_Tp${TP}_bscaI_k${K}ms/best_G.pt:${K} --ckpt bsc_k${K}=./runs/v9b_Tp${TP}_bsc_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_bscaI_${S}.json > ./runs/eval_v9b_Tp${TP}_bscaI_${S}.log 2>&1
  python -u compare_models.py --data_root $D --split $S --ks 1 2 16 --pair "BSCAI=./runs/v9b_Tp${TP}_bscaI_k{K}ms/best_G.pt" \
    --pair "BSC=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/compare_bscaI_bsc_${S}.json > ./runs/compare_bscaI_bsc_${S}.log 2>&1
done
python -u plot_antenna_error.py --ks 1 16 --out figures/antenna_error_map_v9b_bscaI.png > ./runs/antenna_error_bscaI.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v9b_Tp${TP}_bscaI_k*ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_Tp${TP}_bscaI_*.log ./runs/compare_bscaI_bsc_*.log ./runs/antenna_error_bscaI.log
echo done
