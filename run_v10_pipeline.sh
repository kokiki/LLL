#!/usr/bin/env bash
# v10 (기하 일관: nadir 지향 배열, 단말 cross-track 위치 -> 방위각·Doppler·Doppler 변화율 동시 계산) 전체 생성
#   -> 점검 -> BSC-UNet-I (개선 학습 설정) 8 horizon 학습 (2개 동시) -> test·guard 평가 (baseline 7종, 누수 점검), 일반화 그림
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
"/c/Program Files/MATLAB/R2026b/bin/matlab" -batch "generate_ntn_csi_v10_fdd_dl" > gen_v10_full.log 2>&1
echo "[$(date)] generation done: $(ls $D/*/episode_*.mat | wc -l) episodes"
python -u validate_v10.py > ./runs/validate_v10.log 2>&1; cat ./runs/validate_v10.log | grep -v Warn
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
run() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp 20 $IMP --out_dir ./runs/v10_Tp20_bscI_k${K}ms > ./runs/train_v10_Tp20_bscI_k${K}ms.log 2>&1
done; }
run 1 4 12 24 &
run 2 8 16 32 &
wait
echo "[$(date)] training done"
ck() { for K in "$@"; do echo -n " --ckpt BSCI_k${K}=./runs/v10_Tp20_bscI_k${K}ms/best_G.pt:${K}"; done; }
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 1 2 4 8 12 16) --leak_check --out ./runs/eval_v10_bscI_short_${S}.json > ./runs/eval_v10_bscI_short_${S}.log 2>&1
  python -u evaluate.py --data_root $D --split $S --Tp 20 $(ck 24 32) --leak_check --out ./runs/eval_v10_bscI_long_${S}.json > ./runs/eval_v10_bscI_long_${S}.log 2>&1
done
python -u plot_generalization.py --data_root $D --ckpt "runs/v10_Tp20_bscI_k{K}ms/best_G.pt" --out figures/generalization_nmse_vs_elevation_v10_bscI.png > ./runs/plot_generalization_v10.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v10_Tp20_bscI_k*ms.log
grep -aE "Saved|Traceback" ./runs/eval_v10_bscI_*.log
echo done
