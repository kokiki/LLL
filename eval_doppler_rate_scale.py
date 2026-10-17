"""
eval_doppler_rate_scale.py -- Doppler 변화율 스케일 κ의 영향: v9b(κ = 0.155)와 실제 변화율(κ = 1) 비교
  저장된 v9b 채널에 빠진 2차 위상 exp(jπ(ḟ_raw − ḟ)t²) (ḟ_raw = ḟ/κ, t = 관측 시각, 모든 안테나·부반송파 공통)를 곱해
  κ = 1 데이터를 재생성 없이 만든다 (관측 슬롯 안 시각 오프셋 δ의 영향은 선형 위상 항이라 로더의 회전 제거로 사라짐).
  baseline은 각 조건의 train/val로 다시 적합, BSC-UNet-I는 κ = 0.155로 학습한 가중치 그대로 (κ = 1에서는 학습 조건 불일치).
  출력: runs/eval_doppler_rate_scale.json
"""
import json

import numpy as np
import torch

from baselines import Baselines
from compare_models import load
from data import Split, Windows, from_model_out, to_model_io

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
TP, TF, KS = 20, 16, [1, 4, 16]
KAPPA = 0.1550556013060905
BASES = ["Persistence", "LinReg_eig", "Poly2_blind", "ManifoldLinReg", "ManifoldPoly2"]
C128 = torch.complex128


def to_raw_rate(sp):
    """sp.X에 exp(jπ(ḟ/κ − ḟ)t²)를 곱해 κ = 1 조건으로 바꾼다."""
    fdd = torch.as_tensor(sp.meta["fDdot_sat"].astype(np.float64), device=sp.X.device)          # (E,) 적용된 ḟ
    t = torch.arange(sp.T, device=sp.X.device, dtype=torch.float64) * sp.dt_ms * 1e-3              # (T,)
    ph = torch.pi * (fdd * (1 / KAPPA - 1))[:, None] * t[None, :] ** 2                               # (E, T)
    sp.X = sp.X * torch.exp(1j * ph).to(sp.X.dtype)[:, None, :, None]


@torch.no_grad()
def evaluate(tr, va, te):
    bl = Baselines(TP, KS, 1.0).fit(Windows(tr, TP, TF), Windows(va, TP, TF))
    G = {k: load(f"runs/v9b_Tp20_bscI_k{k}ms/best_G.pt", TP, "cuda") for k in KS}
    names = ["BSC-UNet-I"] + BASES
    num = {m: np.zeros(len(KS)) for m in names}; den = np.zeros(len(KS))
    for d in Windows(te, TP, TF).eval_batches(16):
        B, S = d["hist_n"].shape[:2]
        bp = bl.predict(d)
        for j, k in enumerate(KS):
            Y = d["fut"][:, :, k - 1].to(C128)
            den[j] += float((Y.abs() ** 2).sum())
            p = from_model_out(G[k](to_model_io(d, k)[0]), B, S).to(C128)
            num["BSC-UNet-I"][j] += float(((p - Y).abs() ** 2).sum())
            for m in BASES:
                num[m][j] += float(((bp[m][:, :, j] - Y).abs() ** 2).sum())
    return {m: {str(k): 10 * np.log10(num[m][j] / den[j]) for j, k in enumerate(KS)} for m in names}


def main():
    out = {}
    for label, raw in (("kappa=0.155 (v9b)", False), ("kappa=1 (실제 변화율)", True)):
        sps = [Split(D, s) for s in ("train", "val", "test")]
        if raw:
            for sp in sps:
                to_raw_rate(sp)
        out[label] = evaluate(*sps)
        print(f"\n=== {label} : test NMSE [dB] ===", flush=True)
        for m, r in out[label].items():
            print(f"  {m:16s}" + "".join(f"  k={k}: {r[str(k)]:7.2f}" for k in KS), flush=True)
    print("\n=== κ = 1 − κ = 0.155 (양수 = 실제 변화율에서 나빠짐) ===")
    a, b = out["kappa=0.155 (v9b)"], out["kappa=1 (실제 변화율)"]
    for m in a:
        print(f"  {m:16s}" + "".join(f"  k={k}: {b[m][str(k)] - a[m][str(k)]:+7.2f}" for k in KS))
    json.dump(out, open("runs/eval_doppler_rate_scale.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
