#!/usr/bin/env bash
# BSC-UNet-I + 대칭 출력층 (k=16) -> BSC-UNet-I와 paired 비교 (test, guard), 안테나별 오차 지도
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
IMP="--lr_sched cosine --cosine_epochs 80 --ema 0.999 --beta1 0.9 --grad_clip 1.0 --patience 20 --epochs 100"
python -u train_beam.py --data_root $D --horizon_ms 16 --Tp 20 $IMP --sym_out --out_dir ./runs/v9b_Tp20_bscIS_k16ms > ./runs/train_v9b_Tp20_bscIS_k16ms.log 2>&1
for S in test guard; do
  python -u compare_models.py --data_root $D --split $S --ks 16 --pair "BSCIS=./runs/v9b_Tp20_bscIS_k{K}ms/best_G.pt" \
    --pair "BSCI=./runs/v9b_Tp20_bscI_k{K}ms/best_G.pt" --out ./runs/compare_bscIS_bscI_${S}.json > ./runs/compare_bscIS_bscI_${S}.log 2>&1
done
python -u plot_antenna_error.py --ks 16 --out figures/antenna_error_map_v9b_bscIS.png > ./runs/antenna_error_bscIS.log 2>&1
grep -HE "Test NMSE|early|Traceback" ./runs/train_v9b_Tp20_bscIS_k16ms.log
grep -aE "k=16" ./runs/compare_bscIS_bscI_*.log
grep -E "k=16" ./runs/antenna_error_bscIS.log
echo done
