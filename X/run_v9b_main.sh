#!/usr/bin/env bash
# v9b 생성 완료 후: (1) 검증 -> (2) BSC-UNet 6 horizon 학습 [+ (4) DL baseline 4종 병렬] -> (3) test+guard 평가, 일반화 그림
#                  -> (4) 진단(오라클·SNR), 배열 오차 견고성 + 온라인 보정
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
TP=20
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
until [ "$(grep -c '/400\]' gen_v9b_full.log 2>/dev/null)" -ge 400 ] && ! tasklist | grep -qE "^MATLAB"; do sleep 120; done
echo "[$(date)] generation done: $(ls $D/*/episode_*.mat | wc -l) episodes"
python -u validate_v9b.py > ./runs/validate_v9b.log 2>&1; tail -1 ./runs/validate_v9b.log
run() { for K in "$@"; do
  python -u train_beam.py --data_root $D --horizon_ms $K --Tp $TP --out_dir ./runs/v9b_Tp${TP}_bsc_k${K}ms > ./runs/train_v9b_Tp${TP}_bsc_k${K}ms.log 2>&1
done; }
run 1 4 12 & P1=$!
run 2 8 16 & P2=$!
python -u train_baseline_dl.py --data_root $D --models lstm,gru,lstmcnn,transformer --horizons 1,2,4,8,12,16 --tp $TP \
  --phase_aug --epochs 100 --patience 15 --save_ckpt --out_dir ./runs/v9b_dl_baselines > ./runs/train_v9b_dl_baselines.log 2>&1 & P3=$!
wait $P1 $P2
echo "[$(date)] BSC training done"
CK=""; for K in 1 2 4 8 12 16; do CK="$CK --ckpt BSC_k${K}=./runs/v9b_Tp${TP}_bsc_k${K}ms/best_G.pt:${K}"; done
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9b_Tp${TP}_${S}.json > ./runs/eval_v9b_Tp${TP}_${S}.log 2>&1
done
python -u plot_generalization.py --data_root $D --ckpt "runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out figures/generalization_nmse_vs_elevation_v9b.png > ./runs/plot_generalization_v9b.log 2>&1
for S in test guard; do
  python -u diagnose.py --data_root $D --split $S $CK --out ./runs/diag_v9b_Tp${TP}_${S}.json > ./runs/diag_v9b_Tp${TP}_${S}.log 2>&1
done
python -u robust_calib.py --data_root $D --split test --pair "BSC=./runs/v9b_Tp${TP}_bsc_k{K}ms/best_G.pt" --out ./runs/robust_calib_v9b_test.json > ./runs/robust_calib_v9b_test.log 2>&1
echo "[$(date)] analysis done"
wait $P3
echo "[$(date)] DL baselines done"
grep -HE "Test NMSE|Traceback" ./runs/train_v9b_Tp${TP}_bsc_k*ms.log
grep -aHE "Saved|Traceback|saved" ./runs/eval_v9b_*.log ./runs/diag_v9b_*.log ./runs/robust_calib_v9b_test.log ./runs/plot_generalization_v9b.log
tail -12 ./runs/train_v9b_dl_baselines.log
echo done
