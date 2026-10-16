#!/usr/bin/env bash
# BSC-UNet-I k=24, 32 ms (LEO 투명형 최대 왕복 지연 ~26 ms 커버) 동시 학습 -> test+guard 평가 (baseline 7종, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 24 32; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --out_dir ./runs/v9b_Tp20_bscI_k${K}ms > ./runs/train_v9b_Tp20_bscI_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
CK="--ckpt BSCI_k24=./runs/v9b_Tp20_bscI_k24ms/best_G.pt:24 --ckpt BSCI_k32=./runs/v9b_Tp20_bscI_k32ms/best_G.pt:32"
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $CK --leak_check --out ./runs/eval_v9b_Tp20_bscI_long_${S}.json > ./runs/eval_v9b_Tp20_bscI_long_${S}.log 2>&1
done
grep -HE "Test NMSE|early|Traceback|windows:" ./runs/train_v9b_Tp20_bscI_k24ms.log ./runs/train_v9b_Tp20_bscI_k32ms.log
grep -aE "Saved|Traceback" ./runs/eval_v9b_Tp20_bscI_long_*.log
echo done
