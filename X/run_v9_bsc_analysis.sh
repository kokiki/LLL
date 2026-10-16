#!/usr/bin/env bash
# BSC-UNet 분석: (1) BSC vs LSC paired + (2) 배열 보정 오차 견고성 (test, guard), (3) 진단 (오라클 참고값·SNR별)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
export OMP_NUM_THREADS=2
for S in test guard; do
  R=""; [ $S = test ] && R="--robust"
  python -u compare_models.py --data_root $D --split $S --pair "BSC=./runs/v9_Tp20_beam_k{K}ms/best_G.pt" \
    --pair "LSC=./runs/v9_Tp20_k{K}ms/best_G.pt" $R --out ./runs/compare_bsc_lsc_${S}.json > ./runs/compare_bsc_lsc_${S}.log 2>&1
done
CK=""; for K in 1 2 4 8 12 16; do CK="$CK --ckpt BSC_k${K}=./runs/v9_Tp20_beam_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u diagnose.py --data_root $D --split $S $CK --out ./runs/diag_v9_Tp20_bsc_${S}.json > ./runs/diag_v9_Tp20_bsc_${S}.log 2>&1
done
grep -aE "Saved|Traceback|Error" ./runs/compare_bsc_lsc_*.log ./runs/diag_v9_Tp20_bsc_*.log
echo done
