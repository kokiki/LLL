"""
analyze_bias.py -- 예측 오차의 편향(bias)과 흔들림(jitter) 분해: BSC-UNet vs Poly2 (v9b test, k=16)
  rel = 예측 / 실제 (복소). |rel| = 크기 비율 (1 = 정확), angle(rel) = 위상 오차.
  중앙값·평균 = 편향, 표준편차 = 흔들림. 안테나 1개(그래프와 같음)와 전체(256 안테나 x 12 subcarrier)를 따로 계산.
"""
import numpy as np
import torch

from baselines import Baselines
from compare_models import load
from data import Split, Windows, from_model_out, to_model_io

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
Tp, Tf, K = 20, 16, 16
C = torch.complex128


def main():
    tr, va, te = (Split(D, s) for s in ("train", "val", "test"))
    Wte = Windows(te, Tp, Tf)
    bl = Baselines(Tp, [K], 1.0).fit(Windows(tr, Tp, Tf), Windows(va, Tp, Tf))
    G = load(f"runs/v9b_Tp20_bsc_k{K}ms/best_G.pt", Tp, "cuda")
    nw = te.T - Tp - Tf + 1
    for name in ("episode_00023.mat", "episode_00061.mat"):
        e = te.names.index(name)
        rows = torch.arange(e * nw, (e + 1) * nw, device="cuda")
        g = torch.Generator(device="cuda"); g.manual_seed(0)
        with torch.no_grad():
            d = Wte.batch(rows, g); B, S = d["hist_n"].shape[:2]
            Y = d["fut"][:, :, K - 1].to(C)
            P = {"BSC": from_model_out(G(to_model_io(d, K)[0]), B, S).to(C), "Poly2": bl.predict(d)["Poly2_blind"][:, :, 0]}
        print(f"\n{name} ({'NLOS' if te.nlos[e] else 'LOS'}), k={K} ms")
        for scope, sl in (("안테나 1개 (그래프)", (slice(None), 0, 0)), ("전체 256안테나x12sc", (slice(None),) * 3)):
            for m, p in P.items():
                y, q = Y[sl].flatten(), p[sl].flatten()
                nm = 10 * np.log10(((q - y).abs() ** 2).sum().item() / (y.abs() ** 2).sum().item())
                rel = q / y; ph = torch.angle(rel); mag = rel.abs()
                print(f"  {scope:20s} {m:5s}: NMSE {nm:6.1f} dB | 크기 비율 중앙값 {mag.median().item():.3f} "
                      f"(표준편차 {mag.std().item():.3f}) | 위상 오차 평균 {np.degrees(ph.mean().item()):+5.1f}° "
                      f"표준편차 {np.degrees(ph.std().item()):5.1f}°", flush=True)


if __name__ == "__main__":
    main()
