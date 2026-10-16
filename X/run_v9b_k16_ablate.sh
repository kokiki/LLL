#!/usr/bin/env bash
# k=16 원인 분리: (2) BSC-UNet-AD = 빔 정렬 + 빔 방향 채널 (개선 설정), (3) BSC-UNet + 개선 설정 (구조 그대로) -> 동시 학습
#   -> BSC-UNet / BSC-UNet+개선 / BSC-UNet-A+개선 / BSC-UNet-AD 비교 (test, guard), paired, 안테나별 오차 지도
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
python -u train_beam.py --data_root $D --horizon_ms 16 --Tp $TP --aligned_dir $IMP --out_dir ./runs/v9b_Tp${TP}_bscaD_k16ms > ./runs/train_v9b_Tp${TP}_bscaD_k16ms.log 2>&1 &
python -u train_beam.py --data_root $D --horizon_ms 16 --Tp $TP $IMP --out_dir ./runs/v9b_Tp${TP}_bscI_k16ms > ./runs/train_v9b_Tp${TP}_bscI_k16ms.log 2>&1 &
wait
echo "[$(date)] training done"
CK=""; for M in bsc bscI bscaI bscaD; do CK="$CK --ckpt ${M}_k16=./runs/v9b_Tp${TP}_${M}_k16ms/best_G.pt:16"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --out ./runs/eval_v9b_Tp${TP}_k16ablate_${S}.json > ./runs/eval_v9b_Tp${TP}_k16ablate_${S}.log 2>&1
  for M in bscI bscaD; do
    python -u compare_models.py --data_root $D --split $S --ks 16 --pair "${M}=./runs/v9b_Tp${TP}_${M}_k{K}ms/best_G.pt" \
      --pair "bsc=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/compare_${M}_bsc_k16_${S}.json > ./runs/compare_${M}_bsc_k16_${S}.log 2>&1
  done
done
python -u plot_antenna_error.py --ks 16 --out figures/antenna_error_map_v9b_k16ablate.png > ./runs/antenna_error_k16ablate.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v9b_Tp${TP}_bscaD_k16ms.log ./runs/train_v9b_Tp${TP}_bscI_k16ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_Tp${TP}_k16ablate_*.log ./runs/compare_*_bsc_k16_*.log ./runs/antenna_error_k16ablate.log
echo done
