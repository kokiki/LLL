#!/usr/bin/env bash
# v9 (FDD DL, 1 ms 관측, 방위각 고정, 에피소드 단위 블로키지, block split) 전체 파이프라인
#   제안 기법: 선형 시간 지름길 + 비선형 UNet 보정, MSE 손실, 위상 증강, history Tp = 20 (20 ms)
#   1) 제안 기법 학습: k = 16, 8, 4 ms
#   2) 평가: test(개발용) + guard(확인용, 학습/선택에 미사용) -- 제안 기법 + baseline 5종, paired bootstrap, 누수 점검
#   (ablation은 추후: 아래 주석 참고)
set -e
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9
TP=20
export OMP_NUM_THREADS=2
for K in 16 8 4; do
  python -u train.py --data_root $D --horizon_ms $K --Tp $TP --out_dir ./runs/v9_Tp${TP}_k${K}ms > ./runs/train_v9_Tp${TP}_k${K}ms.log 2>&1
done
CK="--ckpt Prop_k16=./runs/v9_Tp${TP}_k16ms/best_G.pt:16 --ckpt Prop_k8=./runs/v9_Tp${TP}_k8ms/best_G.pt:8 --ckpt Prop_k4=./runs/v9_Tp${TP}_k4ms/best_G.pt:4"
for S in test guard; do
  python -u evaluate.py --data_root $D --split $S --Tp $TP $CK --leak_check --out ./runs/eval_v9_Tp${TP}_${S}.json > ./runs/eval_v9_Tp${TP}_${S}.log 2>&1
done
echo "done"

# ---- ablation (추후 실행) ----
# python -u train.py --data_root $D --horizon_ms 16 --Tp $TP --no_shortcut --out_dir ./runs/v9_Tp${TP}_k16ms_noSC   # 선형 지름길 제거
# python -u train.py --data_root $D --horizon_ms 16 --Tp $TP --rcg_cycle  --out_dir ./runs/v9_Tp${TP}_k16ms_rcg    # RCG cycle 추가
# 평가 시 --ckpt Abl_noSC_k16=./runs/v9_Tp${TP}_k16ms_noSC/best_G.pt:16 --ckpt Abl_RCG_k16=./runs/v9_Tp${TP}_k16ms_rcg/best_G.pt:16 추가
