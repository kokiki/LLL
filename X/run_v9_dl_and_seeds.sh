#!/usr/bin/env bash
# (B) DL baseline 4종 x 6 horizon: k=4,8 학습이 끝나면 시작 (제안 기법 순차 학습과 GPU 2개 작업 병렬)
#     제안 기법과 같은 조건: Tp=20, 위상 증강, 100 epoch / patience 15, 같은 평가 시드
# (C) 제안 기법 seed 반복 (43, 44): run_v9_rest2.sh(k=1,2,12,16 + 평가)가 끝나면 시작
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
TP=20
export OMP_NUM_THREADS=2
cd "$(dirname "$0")"
until [ -f ./runs/v9_Tp${TP}_k8ms/result.json ] && [ -f ./runs/v9_Tp${TP}_k4ms/result.json ]; do sleep 60; done
(
  python -u train_baseline_dl.py --data_root $D --models lstm,gru,lstmcnn,transformer --horizons 1,2,4,8,12,16 \
    --tp $TP --phase_aug --epochs 100 --patience 15 --save_ckpt --out_dir ./runs/v9_dl_baselines \
    > ./runs/train_v9_dl_baselines.log 2>&1
) &
until [ -f ./runs/eval_v9_Tp${TP}_guard.json ]; do sleep 60; done
for SEED in 43 44; do
  for K in 4 8 16 1 2 12; do
    python -u train.py --data_root $D --horizon_ms $K --Tp $TP --seed $SEED --out_dir ./runs/v9_Tp${TP}_k${K}ms_s${SEED} \
      > ./runs/train_v9_Tp${TP}_k${K}ms_s${SEED}.log 2>&1
  done
done
wait
CK=""
for K in 1 2 4 8 12 16; do for SEED in 43 44; do CK="$CK --ckpt Prop_k${K}_s${SEED}=./runs/v9_Tp${TP}_k${K}ms_s${SEED}/best_G.pt:${K}"; done; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --n_boot 2000 --out ./runs/eval_v9_Tp${TP}_seeds_${S}.json > ./runs/eval_v9_Tp${TP}_seeds_${S}.log 2>&1
done
grep -HE "Test NMSE" ./runs/train_v9_Tp${TP}_k*ms_s*.log
tail -20 ./runs/train_v9_dl_baselines.log
echo done
