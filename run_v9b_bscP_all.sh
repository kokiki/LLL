#!/usr/bin/env bash
# 최종 모델 BSC-UNet-P (κ = 0.155, v9b) 나머지 horizon 학습 (k=1, 16은 run_v9b_bscP.sh에서 완료)
#   -> 8 horizon test·guard 평가 (baseline 8종 + ablation용 BSC-UNet-I, 누수 점검) -> 방위각 일반화 시험
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
run() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --poly_shortcut \
    --out_dir ./runs/v9b_Tp20_bscP_k${K}ms > ./runs/train_v9b_Tp20_bscP_k${K}ms.log 2>&1
done; }
run 2 12 & run 4 24 & run 8 32 &
wait
echo "[$(date)] training done"
bscI() { case $1 in 1|2|4) echo "./runs/v9b_Tp20_bscI_lrfix_k$1ms/best_G.pt";; *) echo "./runs/v9b_Tp20_bscI_k$1ms/best_G.pt";; esac; }
ck() { for K in "$@"; do echo -n " --ckpt bscP_k${K}=./runs/v9b_Tp20_bscP_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=$(bscI $K):${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 1 2 4 8 12 16) --leak_check --out ./runs/eval_v9b_final_short_${S}.json > ./runs/eval_v9b_final_short_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 24 32) --leak_check --out ./runs/eval_v9b_final_long_${S}.json > ./runs/eval_v9b_final_long_${S}.log 2>&1
done
echo "[$(date)] eval done"
python -u eval_bearing_final.py > ./runs/eval_bearing_v9b_final.log 2>&1
echo "[$(date)] bearing done"
