#!/usr/bin/env bash
# ManifoldCombo(선형 + 2차 외삽 조합, UNet 없음) baseline 추가 후 최종 평가 재실행: κ = 0.155 (k = 1–16), κ = 1 (k = 1, 16)
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
bscI() { case $1 in 1|2|4) echo "./runs/v9b_Tp20_bscI_lrfix_k$1ms/best_G.pt";; *) echo "./runs/v9b_Tp20_bscI_k$1ms/best_G.pt";; esac; }
ck() { for K in "$@"; do echo -n " --ckpt bscP_k${K}=./runs/v9b_Tp20_bscP_k${K}ms/best_G.pt:${K} --ckpt bscI_k${K}=$(bscI $K):${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 1 2 4 8 12 16) --leak_check --out ./runs/eval_v9b_final_short_${S}.json > ./runs/eval_v9b_final_short_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 --leak_check --doppler_rate_kappa 1 \
    --ckpt bscP_k1=./runs/v9b_Tp20_bscP_kappa1_k1ms/best_G.pt:1 --ckpt bscP_k16=./runs/v9b_Tp20_bscP_kappa1_k16ms/best_G.pt:16 \
    --ckpt bscI_k1=./runs/v9b_Tp20_bscI_kappa1_k1ms/best_G.pt:1 --ckpt bscI_k16=./runs/v9b_Tp20_bscI_kappa1_k16ms/best_G.pt:16 \
    --out ./runs/eval_v9b_bscP_kappa1_${S}.json > ./runs/eval_v9b_bscP_kappa1_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 --leak_check \
    --ckpt bscP_k1=./runs/v9b_Tp20_bscP_k1ms/best_G.pt:1 --ckpt bscP_k16=./runs/v9b_Tp20_bscP_k16ms/best_G.pt:16 \
    --ckpt bscI_k1=./runs/v9b_Tp20_bscI_lrfix_k1ms/best_G.pt:1 --ckpt bscI_k16=./runs/v9b_Tp20_bscI_k16ms/best_G.pt:16 \
    --out ./runs/eval_v9b_bscP_${S}.json > ./runs/eval_v9b_bscP_${S}.log 2>&1
done
echo "[$(date)] eval done"
