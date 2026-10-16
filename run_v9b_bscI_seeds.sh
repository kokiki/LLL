#!/usr/bin/env bash
# BSC-UNet-I seed 반복: seed 43, 44 x 8 horizon (1,2,4,8,12,16,24,32 ms), 2개 동시
#   -> seed 42(기존)와 함께 3 seed 평가 (test, guard). 윈도우 조건을 기존 결과와 맞추려고 k<=16 / k=24,32 따로 평가
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
run() { SEED=$1; shift; for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --seed $SEED --out_dir ./runs/v9b_Tp20_bscI_k${K}ms_s${SEED} \
    > ./runs/train_v9b_Tp20_bscI_k${K}ms_s${SEED}.log 2>&1
done; }
run 43 1 2 4 8 12 16 24 32 &
run 44 1 2 4 8 12 16 24 32 &
wait
echo "[$(date)] seed training done"
ck() { for K in "$@"; do echo -n " --ckpt BSCI_k${K}_s42=./runs/v9b_Tp20_bscI_k${K}ms/best_G.pt:${K}"
  for SEED in 43 44; do echo -n " --ckpt BSCI_k${K}_s${SEED}=./runs/v9b_Tp20_bscI_k${K}ms_s${SEED}/best_G.pt:${K}"; done; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 1 2 4 8 12 16) --n_boot 2000 --out ./runs/eval_v9b_bscI_seeds_short_${S}.json > ./runs/eval_v9b_bscI_seeds_short_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 24 32) --n_boot 2000 --out ./runs/eval_v9b_bscI_seeds_long_${S}.json > ./runs/eval_v9b_bscI_seeds_long_${S}.log 2>&1
done
grep -HE "Test NMSE|Traceback" ./runs/train_v9b_Tp20_bscI_k*ms_s4*.log
grep -aE "Saved|Traceback" ./runs/eval_v9b_bscI_seeds_*.log
echo done
