#!/usr/bin/env bash
# BSC-UNet-A (빔 정렬 좌표 UNet, replicate 패딩) 6 horizon 학습 (2개 동시) -> 기존 BSC-UNet과 test+guard 비교, 안테나별 오차 지도
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
run() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp $TP --aligned --out_dir ./runs/v9b_Tp${TP}_bsca_k${K}ms > ./runs/train_v9b_Tp${TP}_bsca_k${K}ms.log 2>&1
done; }
run 1 4 12 &
run 2 8 16 &
wait
echo "[$(date)] BSC-A training done"
CK=""; for K in 1 2 4 8 12 16; do CK="$CK --ckpt BSCA_k${K}=./runs/v9b_Tp${TP}_bsca_k${K}ms/best_G.pt:${K} --ckpt BSC_k${K}=./runs/v9b_Tp${TP}_bsc_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_bsca_${S}.json > ./runs/eval_v9b_Tp${TP}_bsca_${S}.log 2>&1
  python -u compare_models.py --data_root $D --split $S --pair "BSCA=./runs/v9b_Tp${TP}_bsca_k{K}ms/best_G.pt" \
    --pair "BSC=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/compare_bsca_bsc_${S}.json > ./runs/compare_bsca_bsc_${S}.log 2>&1
done
python -u plot_antenna_error.py > ./runs/antenna_error_bsca.log 2>&1
grep -HE "Test NMSE|Traceback" ./runs/train_v9b_Tp${TP}_bsca_k*ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_Tp${TP}_bsca_*.log ./runs/compare_bsca_bsc_*.log ./runs/antenna_error_bsca.log
echo done
