"""
plot_antenna_error.py -- 안테나별 NMSE 지도 (16x16 배열, v9b test 전체): UNet 합성곱의 배열 가장자리 효과 확인
  NMSE(안테나 i) = Σ_{윈도우, subcarrier} |p_i - y_i|² / Σ |y_i|²
  가장자리 = 배열 테두리 1줄(60개), 안쪽 = 나머지(196개). 모서리 4개는 따로 표시.
출력: figures/antenna_error_map_v9b.png
"""
import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import Baselines
from compare_models import load
from data import Split, Windows, from_model_out, to_model_io

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
Tp, Tf = 20, 16
C = torch.complex128
METHODS = {"BSC": "BSC-UNet", "BSCI": "BSC-UNet + 개선 학습", "BSCA": "BSC-UNet-A (빔 정렬)", "BSCAI": "BSC-UNet-A + 개선 학습",
           "BSCAD": "BSC-UNet-AD (빔 정렬+방향)", "BSCIS": "BSC-UNet-I + 대칭 출력층", "BSCIR": "BSC-UNet-I + replicate 패딩",
           "Poly2_blind": "Poly2", "ManifoldLinReg": "ManifoldLinReg"}
CKPT = {"BSC": "runs/v9b_Tp20_bsc_k{K}ms/best_G.pt", "BSCA": "runs/v9b_Tp20_bsca_k{K}ms/best_G.pt",
        "BSCAI": "runs/v9b_Tp20_bscaI_k{K}ms/best_G.pt", "BSCI": "runs/v9b_Tp20_bscI_k{K}ms/best_G.pt",
        "BSCAD": "runs/v9b_Tp20_bscaD_k{K}ms/best_G.pt", "BSCIS": "runs/v9b_Tp20_bscIS_k{K}ms/best_G.pt",
        "BSCIR": "runs/v9b_Tp20_bscIR_k{K}ms/best_G.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--out", default="figures/antenna_error_map_v9b.png")
    a = ap.parse_args()
    KS = tuple(a.ks)
    tr, va, te = (Split(D, s) for s in ("train", "val", "test"))
    Wte = Windows(te, Tp, Tf)
    bl = Baselines(Tp, list(KS), 1.0).fit(Windows(tr, Tp, Tf), Windows(va, Tp, Tf))
    for m in list(CKPT):
        if not all(os.path.exists(CKPT[m].replace("{K}", str(k))) for k in KS):
            METHODS.pop(m); CKPT.pop(m)
    G = {m: {k: load(CKPT[m].replace("{K}", str(k)), Tp, "cuda") for k in KS} for m in CKPT}
    num = {m: {k: torch.zeros(256, dtype=torch.float64, device="cuda") for k in KS} for m in METHODS}
    den = {k: torch.zeros(256, dtype=torch.float64, device="cuda") for k in KS}
    with torch.no_grad():
        for d in Wte.eval_batches(16):
            B, S = d["hist_n"].shape[:2]
            bp = bl.predict(d)
            for j, k in enumerate(KS):
                Y = d["fut"][:, :, k - 1].to(C)
                den[k] += (Y.abs() ** 2).sum(dim=(0, 1))
                P = {"Poly2_blind": bp["Poly2_blind"][:, :, j], "ManifoldLinReg": bp["ManifoldLinReg"][:, :, j]}
                for m in G:
                    P[m] = from_model_out(G[m][k](to_model_io(d, k)[0]), B, S).to(C)
                for m in METHODS:
                    num[m][k] += ((P[m] - Y).abs() ** 2).sum(dim=(0, 1))
    edge = np.zeros((16, 16), bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    corner = np.zeros((16, 16), bool); corner[[0, 0, -1, -1], [0, -1, 0, -1]] = True
    fig, ax = plt.subplots(len(KS), len(METHODS), figsize=(5.4 * len(METHODS), 5.3 * len(KS)), squeeze=False)
    for r, k in enumerate(KS):
        maps = {m: (10 * torch.log10(num[m][k] / den[k])).reshape(16, 16).cpu().numpy() for m in METHODS}
        lo = min(v.min() for v in maps.values()); hi = max(v.max() for v in maps.values())
        for c, m in enumerate(METHODS):
            v = maps[m]
            db = lambda mask: 10 * np.log10(num[m][k].cpu().numpy()[mask.ravel()].sum() / den[k].cpu().numpy()[mask.ravel()].sum())
            im = ax[r, c].imshow(v, cmap="viridis_r", vmin=lo, vmax=hi)
            ax[r, c].set_title(f"{METHODS[m]}, k={k} ms\n안쪽 {db(~edge):.1f} / 가장자리 {db(edge & ~corner):.1f} / 모서리 {db(corner):.1f} dB",
                               fontsize=10)
            ax[r, c].set_xlabel("안테나 열 n"); ax[r, c].set_ylabel("안테나 행 m")
            fig.colorbar(im, ax=ax[r, c], fraction=0.046, label="NMSE [dB] (밝을수록 좋음)")
            print(f"k={k} {METHODS[m]:15s}: 안쪽 {db(~edge):.2f}  가장자리 {db(edge & ~corner):.2f}  모서리 {db(corner):.2f}  "
                  f"전체 {10 * np.log10(num[m][k].sum().item() / den[k].sum().item()):.2f} dB", flush=True)
    fig.suptitle("안테나별 NMSE 지도 (v9b test 전체, 16x16 위성 배열) — 같은 행의 색 범위 공통", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=100)
    print("saved")


if __name__ == "__main__":
    main()
