#!/usr/bin/env bash
# BSC-UNet (models_beam.BeamShortcutUNet, train_beam.py) 6개 horizon 학습 (2개 동시) -> LSC-UNet과 함께 test+guard 평가
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
TP=20
export OMP_NUM_THREADS=2
run() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp $TP --out_dir ./runs/v9_Tp${TP}_beam_k${K}ms > ./runs/train_v9_Tp${TP}_beam_k${K}ms.log 2>&1
done; }
run 1 4 12 &
run 2 8 16 &
wait
CK=""
for K in 1 2 4 8 12 16; do CK="$CK --ckpt LSC_k${K}=./runs/v9_Tp${TP}_k${K}ms/best_G.pt:${K} --ckpt BSC_k${K}=./runs/v9_Tp${TP}_beam_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9_Tp${TP}_bsc_${S}.json > ./runs/eval_v9_Tp${TP}_bsc_${S}.log 2>&1
done
grep -HE "Test NMSE|early stopping|Traceback" ./runs/train_v9_Tp${TP}_beam_k*ms.log
grep -aE "Saved|Traceback" ./runs/eval_v9_Tp${TP}_bsc_*.log
echo done
