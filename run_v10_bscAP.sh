#!/usr/bin/env bash
# v10 BSC-UNet-AP (빔 정렬 + Poly2 외삽 빔계수 지름길, models_beam_poly.py) k=1, 16, 32 시험 학습
#   -> test·guard 평가 (baseline 8종 [ManifoldPoly2 포함] + BSC-UNet-AI 참고, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
for K in 1 16 32; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --aligned_poly --out_dir ./runs/v10_Tp20_bscAP_k${K}ms > ./runs/train_v10_Tp20_bscAP_k${K}ms.log 2>&1 &
done
wait
echo "[$(date)] training done"
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 \
    --ckpt bscAP_k1=./runs/v10_Tp20_bscAP_k1ms/best_G.pt:1 --ckpt bscAI_k1=./runs/v10_Tp20_bscAI_k1ms/best_G.pt:1 \
    --ckpt bscAP_k16=./runs/v10_Tp20_bscAP_k16ms/best_G.pt:16 --ckpt bscAI_k16=./runs/v10_Tp20_bscAI_k16ms/best_G.pt:16 \
    --leak_check --out ./runs/eval_v10_bscAP_short_${S}.json > ./runs/eval_v10_bscAP_short_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 \
    --ckpt bscAP_k32=./runs/v10_Tp20_bscAP_k32ms/best_G.pt:32 --ckpt bscAI_k32=./runs/v10_Tp20_bscAI_k32ms/best_G.pt:32 \
    --leak_check --out ./runs/eval_v10_bscAP_long_${S}.json > ./runs/eval_v10_bscAP_long_${S}.log 2>&1
done
echo "[$(date)] eval done"
