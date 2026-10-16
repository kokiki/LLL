#!/usr/bin/env bash
# k=8, k=4(진행 중)가 끝나면 k=1, 2, 12, 16 순차 학습 (k=16은 처음부터 재학습) -> 6개 horizon 전부 test+guard 평가
set -e
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
TP=20
export OMP_NUM_THREADS=2
until [ -f ./runs/v9_Tp${TP}_k8ms/result.json ] && [ -f ./runs/v9_Tp${TP}_k4ms/result.json ]; do sleep 60; done
for K in 1 2 12 16; do
  python -u train.py --data_root $D --horizon_ms $K --Tp $TP --out_dir ./runs/v9_Tp${TP}_k${K}ms > ./runs/train_v9_Tp${TP}_k${K}ms.log 2>&1
done
CK=""
for K in 1 2 4 8 12 16; do CK="$CK --ckpt Prop_k${K}=./runs/v9_Tp${TP}_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9_Tp${TP}_${S}.json > ./runs/eval_v9_Tp${TP}_${S}.log 2>&1
done
grep -HE "Test NMSE|early stopping" ./runs/train_v9_Tp${TP}_k*ms.log
grep -aE "Saved|Traceback" ./runs/eval_v9_Tp${TP}_*.log
echo done
