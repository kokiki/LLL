#!/usr/bin/env bash
# BSC-UNet-I 기준 분석 갱신: 진단(오라클·SNR), 예측 품질 그림, 배열 오차 견고성 + 온라인 보정
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
CK=""; for K in 1 2 4 8 12 16; do CK="$CK --ckpt BSCI_k${K}=./runs/v9b_Tp20_bscI_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u diagnose.py --data_root $D --split $S $CK --out ./runs/diag_v9b_bscI_${S}.json > ./runs/diag_v9b_bscI_${S}.log 2>&1
done
python -u plot_prediction_quality.py --ckpt "runs/v9b_Tp20_bscI_k{K}ms/best_G.pt" --label "BSC-UNet-I" --out figures/prediction_quality_v9b_bscI.png > ./runs/prediction_quality_bscI.log 2>&1
python -u robust_calib.py --data_root $D --split test --pair "BSCI=./runs/v9b_Tp20_bscI_k{K}ms/best_G.pt" --out ./runs/robust_calib_v9b_bscI_test.json > ./runs/robust_calib_v9b_bscI_test.log 2>&1
grep -aHE "Saved|saved|Traceback" ./runs/diag_v9b_bscI_*.log ./runs/prediction_quality_bscI.log ./runs/robust_calib_v9b_bscI_test.log
echo done
