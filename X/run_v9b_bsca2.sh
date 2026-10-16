#!/usr/bin/env bash
# BSC-UNet-A: 현재 설정 k=1,2(진행 중),16 + 개선 학습 설정(cosine lr, EMA 0.999, beta1 0.9, grad clip 1.0, patience 20) k=1,2,16
#   -> BSC-UNet / BSC-UNet-A / BSC-UNet-A 개선 비교 (test, guard), paired, 안테나별 오차 지도
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
tr_cur() { python -u train_beam.py --data_root $D --horizon_ms $1 --Tp $TP --aligned --out_dir ./runs/v9b_Tp${TP}_bsca_k$1ms > ./runs/train_v9b_Tp${TP}_bsca_k$1ms.log 2>&1; }
tr_imp() { python -u train_beam.py --data_root $D --horizon_ms $1 --Tp $TP --aligned $IMP --out_dir ./runs/v9b_Tp${TP}_bscaI_k$1ms > ./runs/train_v9b_Tp${TP}_bscaI_k$1ms.log 2>&1; }
( until [ -f ./runs/v9b_Tp${TP}_bsca_k1ms/result.json ]; do sleep 60; done; tr_imp 1; tr_imp 16 ) &
( until [ -f ./runs/v9b_Tp${TP}_bsca_k2ms/result.json ]; do sleep 60; done; tr_cur 16; tr_imp 2 ) &
wait
echo "[$(date)] training done"
CK=""; for K in 1 2 16; do for M in bsc bsca bscaI; do CK="$CK --ckpt ${M}_k${K}=./runs/v9b_Tp${TP}_${M}_k${K}ms/best_G.pt:${K}"; done; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_bsca2_${S}.json > ./runs/eval_v9b_Tp${TP}_bsca2_${S}.log 2>&1
  python -u compare_models.py --data_root $D --split $S --ks 1 2 16 --pair "BSCA=./runs/v9b_Tp${TP}_bsca_k{K}ms/best_G.pt" \
    --pair "BSC=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/compare_bsca_bsc_${S}.json > ./runs/compare_bsca_bsc_${S}.log 2>&1
  python -u compare_models.py --data_root $D --split $S --ks 1 2 16 --pair "BSCAI=./runs/v9b_Tp${TP}_bscaI_k{K}ms/best_G.pt" \
    --pair "BSCA=./runs/v9b_Tp${TP}_bsca_k{K}ms/best_G.pt" --out ./runs/compare_bscaI_bsca_${S}.json > ./runs/compare_bscaI_bsca_${S}.log 2>&1
done
python -u plot_antenna_error.py --ks 1 16 --out figures/antenna_error_map_v9b_bsca.png > ./runs/antenna_error_bsca.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v9b_Tp${TP}_bsca*_k*ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_Tp${TP}_bsca2_*.log ./runs/compare_bsca*_*.log ./runs/antenna_error_bsca.log
echo done
