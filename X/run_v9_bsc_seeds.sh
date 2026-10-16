#!/usr/bin/env bash
# BSC-UNet seed 반복 (43, 44) x 6 horizon, 2개 동시 -> test+guard 평가 (seed 42 = runs/v9_Tp20_beam_k{K}ms 포함)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
TP=20
export OMP_NUM_THREADS=2
run() { SEED=$1; shift; for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp $TP --seed $SEED --out_dir ./runs/v9_Tp${TP}_beam_k${K}ms_s${SEED} \
    > ./runs/train_v9_Tp${TP}_beam_k${K}ms_s${SEED}.log 2>&1
done; }
run 43 1 2 4 8 12 16 &
run 44 1 2 4 8 12 16 &
wait
CK=""
for K in 1 2 4 8 12 16; do
  CK="$CK --ckpt BSC_k${K}_s42=./runs/v9_Tp${TP}_beam_k${K}ms/best_G.pt:${K}"
  for SEED in 43 44; do CK="$CK --ckpt BSC_k${K}_s${SEED}=./runs/v9_Tp${TP}_beam_k${K}ms_s${SEED}/best_G.pt:${K}"; done
done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --n_boot 2000 --out ./runs/eval_v9_Tp${TP}_bsc_seeds_${S}.json > ./runs/eval_v9_Tp${TP}_bsc_seeds_${S}.log 2>&1
done
grep -HE "Test NMSE|Traceback" ./runs/train_v9_Tp${TP}_beam_k*ms_s4*.log
grep -aE "Saved|Traceback" ./runs/eval_v9_Tp${TP}_bsc_seeds_*.log
echo done
