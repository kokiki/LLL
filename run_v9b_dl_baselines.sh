#!/usr/bin/env bash
# 딥러닝 baseline (κ = 1, T_p = 20, k = 1–16; 제안 모델과 같은 학습 설정, train_baseline_dl.py)
#   "딥러닝만" 학습·평가와 BSC-UNet-P 추론 시간 측정(run_v9b_kappa1_unet_then_bench.sh)이 끝난 뒤 시작
#   1) LSTM+CNN 6개 학습 (4개씩) -> 평가 (test·guard, baseline 9종과 유의성 검정, 누수 점검)
#   2) LSTM, GRU, Transformer 18개 학습 (4개씩) -> 평가
cd "$(dirname "$0")"
D=./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b
export OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
n_prev() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -match 'run_v9b_kappa1_unet_then_bench\.sh' }).Count" | tr -d '\r'; }
while [ "$(n_prev)" != "0" ]; do sleep 120; done
echo "[$(date)] start DL baselines"
train() {   # 인자: 모델 목록 -> (모델, k) 조합을 4개씩 동시 학습
  for M in "$@"; do for K in 1 2 4 8 12 16; do echo "$M $K"; done; done | \
    xargs -P 4 -L 1 bash -c 'python -u train_baseline_dl.py --models $0 --horizons $1 --doppler_rate_kappa 1 > ./runs/train_v9b_Tp20_$0_kappa1_k$1ms.log 2>&1'
}
evaluate() {   # 인자: 출력 태그, 모델 목록
  local TAG=$1; shift; local CK=""
  for M in "$@"; do for K in 1 2 4 8 12 16; do CK="$CK --ckpt ${M}_k${K}=dl:${M}:./runs/v9b_Tp20_${M}_kappa1_k${K}ms/best_G.pt:${K}"; done; done
  for S in test guard; do
    python -u evaluate.py --data_root $D --split $S --Tp 20 --doppler_rate_kappa 1 --leak_check $CK \
      --out ./runs/eval_v9b_kappa1_${TAG}_${S}.json > ./runs/eval_v9b_kappa1_${TAG}_${S}.log 2>&1
  done
}
train lstmcnn
echo "[$(date)] LSTM+CNN training done"
evaluate lstmcnn lstmcnn
echo "[$(date)] LSTM+CNN eval done"
train lstm gru transformer
echo "[$(date)] LSTM/GRU/Transformer training done"
evaluate dlrest lstm gru transformer
echo "[$(date)] LSTM/GRU/Transformer eval done"
