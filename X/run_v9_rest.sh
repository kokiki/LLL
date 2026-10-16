#!/usr/bin/env bash
# run_v9_pipeline.sh 재구성: k16(진행 중) / k8(병렬 시작) -> k16 끝나면 k4 -> 셋 다 끝나면 test+guard 평가
set -e
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
TP=20
export OMP_NUM_THREADS=2
python -u train.py --data_root $D --horizon_ms 8 --Tp $TP --out_dir ./runs/v9_Tp${TP}_k8ms > ./runs/train_v9_Tp${TP}_k8ms.log 2>&1 &
until [ -f ./runs/v9_Tp${TP}_k16ms/result.json ]; do sleep 60; done
python -u train.py --data_root $D --horizon_ms 4 --Tp $TP --out_dir ./runs/v9_Tp${TP}_k4ms > ./runs/train_v9_Tp${TP}_k4ms.log 2>&1
wait
CK="--ckpt Prop_k16=./runs/v9_Tp${TP}_k16ms/best_G.pt:16 --ckpt Prop_k8=./runs/v9_Tp${TP}_k8ms/best_G.pt:8 --ckpt Prop_k4=./runs/v9_Tp${TP}_k4ms/best_G.pt:4"
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9_Tp${TP}_${S}.json > ./runs/eval_v9_Tp${TP}_${S}.log 2>&1
done
grep -hE "Test NMSE|early stopping" ./runs/train_v9_Tp${TP}_k*ms.log
grep -aE "Saved|Traceback" ./runs/eval_v9_Tp${TP}_*.log
echo done
