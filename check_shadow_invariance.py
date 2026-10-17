"""
check_shadow_invariance.py -- shadow fading(에피소드 상수 이득)이 학습·평가 입력에 영향을 주지 않는지 검증
  v9b는 LOS shadow fading을 max(0, ·)로 잘라 감쇠 방향만 반영했다. TR 38.811대로 평균 0 로그정규(양방향,
  σ = 4 dB LOS / 6 dB NLOS) 이득 g_e를 에피소드마다 곱했을 때(관측 채널과 저장된 누적 최댓값 c_e(i) 모두 g_e배),
  로더가 만드는 모델 입력(잡음 history)과 정답이 같은지 비교한다. 같으면 모든 예측 방법의 결과가 같다.
"""
import numpy as np
import torch

from data import Split, Windows

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
for split in ("test", "train"):
    sp = Split(D, split)
    W = Windows(sp, 20, 16)
    ref = [{k: d[k].cpu() for k in ("hist_n", "fut")} for d in W.eval_batches(16, seed=7)]
    rng = np.random.default_rng(0)
    sig = np.where(sp.nlos, 6.0, 4.0)
    g = torch.as_tensor(10 ** (rng.normal(0, sig) / 20), device=sp.X.device)          # 양방향 로그정규 이득
    X0, cm0 = sp.X.clone(), sp.cm.clone()
    sp.X = X0 * g[:, None, None, None].to(X0.dtype); sp.cm = cm0 * g[:, None]
    mx = 0.0
    for d, r in zip(W.eval_batches(16, seed=7), ref):
        for k in ("hist_n", "fut"):
            mx = max(mx, float((d[k].cpu() - r[k]).abs().max() / r[k].abs().max()))
    print(f"{split}: 에피소드 {sp.E}개, 적용 이득 범위 {20*np.log10(g.min().item()):+.1f} ~ {20*np.log10(g.max().item()):+.1f} dB "
          f"-> 모델 입력·정답 최대 상대 차이 {mx:.2e}")
