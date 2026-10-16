#!/usr/bin/env bash
# BSC-UNet-A 개선 설정 k=1,16 학습 완료 대기 -> 기존 BSC-UNet과 test+guard 비교 (paired), 안테나별 오차 지도
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
until [ -f ./runs/v9b_Tp${TP}_bscaI_k1ms/result.json ] && [ -f ./runs/v9b_Tp${TP}_bscaI_k16ms/result.json ]; do sleep 120; done
echo "[$(date)] training done"
CK=""; for K in 1 16; do CK="$CK --ckpt bscaI_k${K}=./runs/v9b_Tp${TP}_bscaI_k${K}ms/best_G.pt:${K} --ckpt bsc_k${K}=./runs/v9b_Tp${TP}_bsc_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_bscaI_${S}.json > ./runs/eval_v9b_Tp${TP}_bscaI_${S}.log 2>&1
  python -u compare_models.py --data_root $D --split $S --ks 1 16 --pair "BSCAI=./runs/v9b_Tp${TP}_bscaI_k{K}ms/best_G.pt" \
    --pair "BSC=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/compare_bscaI_bsc_${S}.json > ./runs/compare_bscaI_bsc_${S}.log 2>&1
done
python -u plot_antenna_error.py --ks 1 16 --out figures/antenna_error_map_v9b_bscaI.png > ./runs/antenna_error_bscaI.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v9b_Tp${TP}_bscaI_k*ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_Tp${TP}_bscaI_*.log ./runs/compare_bscaI_bsc_*.log ./runs/antenna_error_bscaI.log
echo done
