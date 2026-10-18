"""
eval_bearing_final.py -- (최종 모델 BSC-UNet-P + BSC-UNet-I + ManifoldPoly2) 방위각 일반화 검증: test 채널에 열 방향 위상 기울기 e^{jβn}를 곱해 다른 방위각의 단말을 모사
  위성 쪽 채널은 평면파(rank-1)라 방위각 변화 = 배열 전체에 열 방향 위상 기울기를 곱하는 것과 같음 (공간 구조만 변화;
  실제 방위각 변화에 따른 Doppler 기하 변화는 반영하지 않음). 모델·baseline은 방위각 0°(β=0) 데이터로 학습한 그대로.
  출력: β별 전체 NMSE, 안쪽 / 행 가장자리(위·아래) / 열 가장자리(좌·우) NMSE  ->  runs/eval_bearing_v9b_final.json
  BSC-UNet-I는 학습률 수정 모델(k=1, 4)과 기존 모델(k=16, 81 epoch 전 저장으로 동일)을 사용
"""
import argparse
import json

import numpy as np
import torch

from baselines import Baselines
from compare_models import load
from data import Split, Windows, from_model_out, rescale_doppler_rate, to_model_io

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
Tp, Tf, KS = 20, 16, [1, 4, 16]
BETAS = [0.0, 0.5, 1.0, 2.0, 3.0]
C128 = torch.complex128
METHODS = ["BSC-UNet-P", "BSC-UNet-I", "Poly2_blind", "ManifoldLinReg", "ManifoldPoly2", "LinReg_eig"]
MODELS = {"BSC-UNet-P": "runs/v9b_Tp20_bscP_k{k}ms/best_G.pt",
          "BSC-UNet-I": {1: "runs/v9b_Tp20_bscI_lrfix_k1ms/best_G.pt", 4: "runs/v9b_Tp20_bscI_lrfix_k4ms/best_G.pt",
                         16: "runs/v9b_Tp20_bscI_k16ms/best_G.pt"}}


MODELS_KAPPA1 = {"BSC-UNet-P": "runs/v9b_Tp20_bscP_kappa1_k{k}ms/best_G.pt", "BSC-UNet-I": "runs/v9b_Tp20_bscI_kappa1_k{k}ms/best_G.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doppler_rate_kappa", type=float, default=None, help="1 = 실제 Doppler 변화율 조건 (κ = 1로 학습한 모델 사용)")
    a = ap.parse_args()
    global MODELS
    tr, va, te = (Split(D, s) for s in ("train", "val", "test"))
    tag = "final"
    if a.doppler_rate_kappa is not None:
        for sp in (tr, va, te):
            rescale_doppler_rate(sp, a.doppler_rate_kappa)
        MODELS = MODELS_KAPPA1; tag = f"kappa{a.doppler_rate_kappa:g}"
    bl = Baselines(Tp, KS, 1.0).fit(Windows(tr, Tp, Tf), Windows(va, Tp, Tf))
    G = {m: {k: load(p[k] if isinstance(p, dict) else p.format(k=k), Tp, "cuda") for k in KS} for m, p in MODELS.items()}
    n = torch.arange(16, device="cuda", dtype=torch.float64)
    rows = np.zeros((16, 16), bool); rows[[0, -1], :] = True
    cols = np.zeros((16, 16), bool); cols[:, [0, -1]] = True
    masks = {"안쪽": (~rows & ~cols).ravel(), "행 가장자리(위·아래)": (rows & ~cols).ravel(), "열 가장자리(좌·우)": (cols & ~rows).ravel()}
    X0 = te.X.clone()
    out = {}
    for beta in BETAS:
        ramp = torch.exp(1j * beta * n)[None, :].expand(16, 16).reshape(256).to(torch.complex64)   # 안테나 (m, n) C순서 -> 열 n 기울기
        te.X = X0 * ramp
        W = Windows(te, Tp, Tf)
        num = {m: {k: torch.zeros(256, dtype=torch.float64, device="cuda") for k in KS} for m in METHODS}
        den = {k: torch.zeros(256, dtype=torch.float64, device="cuda") for k in KS}
        with torch.no_grad():
            for d in W.eval_batches(16):
                B, S = d["hist_n"].shape[:2]
                bp = bl.predict(d)
                for j, k in enumerate(KS):
                    Y = d["fut"][:, :, k - 1].to(C128)
                    den[k] += (Y.abs() ** 2).sum(dim=(0, 1))
                    x = to_model_io(d, k)[0]
                    P = {m: from_model_out(G[m][k](x), B, S).to(C128) for m in MODELS}
                    for m in METHODS[len(MODELS):]:
                        P[m] = bp[m][:, :, j]
                    for m in METHODS:
                        num[m][k] += ((P[m] - Y).abs() ** 2).sum(dim=(0, 1))
        for k in KS:
            for m in METHODS:
                nn_, dd = num[m][k].cpu().numpy(), den[k].cpu().numpy()
                r = {"전체": 10 * np.log10(nn_.sum() / dd.sum())}
                r.update({g: 10 * np.log10(nn_[mk].sum() / dd[mk].sum()) for g, mk in masks.items()})
                out.setdefault(str(k), {}).setdefault(m, {})[f"{beta:g}"] = r
        print(f"β={beta:g} done", flush=True)
    te.X = X0
    for k in KS:
        print(f"\n===== k={k} ms: 전체 NMSE [dB] vs 열 방향 위상 기울기 β [rad] =====")
        print(f"{'':16s}" + "".join(f"  β={b:<5g}" for b in BETAS))
        for m in METHODS:
            print(f"{m:16s}" + "".join(f"{out[str(k)][m][f'{b:g}']['전체']:9.2f}" for b in BETAS))
        print(f"  {'BSC-UNet-P 위치별':14s}" + "".join(f"  β={b:<5g}" for b in BETAS))
        for g in masks:
            print(f"  {g:16s}" + "".join(f"{out[str(k)]['BSC-UNet-P'][f'{b:g}'][g]:9.2f}" for b in BETAS))
    json.dump(out, open(f"runs/eval_bearing_v9b_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved runs/eval_bearing_v9b_{tag}.json")


if __name__ == "__main__":
    main()
