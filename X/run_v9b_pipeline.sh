#!/usr/bin/env bash
# v9b (타이밍 오프셋 에피소드 고정) 전체: 생성 -> BSC-UNet(제안) + LSC-UNet(ablation) 6 horizon 학습 (2개 동시) -> test+guard 평가
cd "$(dirname "$0")"
"/c/Program Files/MATLAB/R2026b/bin/matlab" -batch "generate_ntn_csi_v9b_fdd_dl" > gen_v9b_full.log 2>&1
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
echo "episodes: $(ls $D/*/episode_*.mat | wc -l)"
TP=20
export OMP_NUM_THREADS=2
run() { for spec in "$@"; do M=${spec%%:*}; K=${spec##*:}
  if [ $M = bsc ]; then S=train_beam.py; O=v9b_Tp${TP}_bsc_k${K}ms; else S=train.py; O=v9b_Tp${TP}_lsc_k${K}ms; fi
  python -u $S --data_root $D --horizon_ms $K --Tp $TP --out_dir ./runs/$O > ./runs/train_$O.log 2>&1
done; }
run bsc:1 bsc:4 bsc:12 lsc:1 lsc:4 lsc:12 &
run bsc:2 bsc:8 bsc:16 lsc:2 lsc:8 lsc:16 &
wait
CK=""
for K in 1 2 4 8 12 16; do CK="$CK --ckpt BSC_k${K}=./runs/v9b_Tp${TP}_bsc_k${K}ms/best_G.pt:${K} --ckpt LSC_k${K}=./runs/v9b_Tp${TP}_lsc_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_${S}.json > ./runs/eval_v9b_Tp${TP}_${S}.log 2>&1
done
grep -HE "Test NMSE|Traceback" ./runs/train_v9b_Tp${TP}_*.log
grep -aE "Saved|Traceback" ./runs/eval_v9b_Tp${TP}_*.log
echo done
