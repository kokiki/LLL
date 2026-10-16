#!/usr/bin/env bash
# v10 BSC-UNet-AI (빔 정렬 입력 + 개선 학습 설정) 나머지 horizon 학습 (k=1, 16은 run_v10_fix.sh에서 완료)
#   -> 8 horizon test·guard 평가 (baseline 7종 + BSC-UNet-I 참고, 누수 점검)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
run() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --aligned --out_dir ./runs/v10_Tp20_bscAI_k${K}ms > ./runs/train_v10_Tp20_bscAI_k${K}ms.log 2>&1
done; }
run 2 12 & run 4 24 & run 8 32 &
wait
echo "[$(date)] training done"
ck() { for K in "$@"; do echo -n " --ckpt bscAI_k${K}=./runs/v10_Tp20_bscAI_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=./runs/v10_Tp20_bscI_k${K}ms/best_G.pt:${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 1 2 4 8 12 16) --leak_check --out ./runs/eval_v10_bscAI_short_${S}.json > ./runs/eval_v10_bscAI_short_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 24 32) --leak_check --out ./runs/eval_v10_bscAI_long_${S}.json > ./runs/eval_v10_bscAI_long_${S}.log 2>&1
done
echo "[$(date)] eval done"
