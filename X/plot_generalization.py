"""
plot_generalization.py -- 일반화 확인 그림: 에피소드별 NMSE vs 고각 (train / val / test / guard)
학습에 쓰지 않은 test·guard·val 에피소드가 같은 고각대의 train 에피소드와 같은 높이에 있으면 일반화 손실이 없다는 뜻.
출력: figures/generalization_nmse_vs_elevation.png
"""
import argparse
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compare_models import load
from data import Split, Windows, to_model_io

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
COL = {"train": "C0", "val": "C2", "test": "C3", "guard": "C1"}
Tp, Tf = 20, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9")
    ap.add_argument("--ckpt", default="runs/v9_Tp20_beam_k{K}ms/best_G.pt", help="{K} -> horizon")
    ap.add_argument("--out", default="figures/generalization_nmse_vs_elevation.png")
    a = ap.parse_args()
    D = a.data_root
    fig, ax = plt.subplots(1, 2, figsize=(17, 5.8))
    for j, k in enumerate((4, 16)):
        G = load(a.ckpt.replace("{K}", str(k)), Tp, "cuda")
        pooled = {}
        for s in COL:
            sp = Split(D, s, fix_timing=False)                          # 모델 학습·평가와 같은 데이터 조건
            W = Windows(sp, Tp, Tf, stride=3 if s == "train" else 1)
            num = np.zeros(sp.E); den = np.zeros(sp.E)
            with torch.no_grad():
                for d in W.eval_batches(16):
                    S = d["hist_n"].shape[1]; x, y = to_model_io(d, k)
                    e = d["e"].repeat_interleave(S).cpu().numpy()
                    np.add.at(num, e, ((G(x) - y) ** 2).sum(dim=(1, 2, 3)).double().cpu().numpy())
                    np.add.at(den, e, (y ** 2).sum(dim=(1, 2, 3)).double().cpu().numpy())
            nm = 10 * np.log10(num / den); el = sp.meta["elevDeg"].astype(float)
            pooled[s] = 10 * np.log10(num.sum() / den.sum())
            jump = Split(D, s).timing_shifts.any(1).cpu().numpy()          # v9 타이밍 기준 점프가 감지된 에피소드
            ax[j].scatter(el[jump], nm[jump], s=150, facecolors="none", edgecolors="k", lw=1.2, zorder=4)
            pooled[s + "_nojump"] = 10 * np.log10(num[~jump].sum() / den[~jump].sum())
            for nl, mk in ((False, "o"), (True, "x")):
                m = sp.nlos == nl
                ax[j].scatter(el[m], nm[m], s=26 if s != "train" else 14, marker=mk, color=COL[s],
                              alpha=0.55 if s == "train" else 0.95, zorder=1 if s == "train" else 3)
        ax[j].set_title(f"BSC-UNet, k={k} ms (검은 원 = v9 타이밍 점프 에피소드)\n전체 NMSE  " +
                        "  ".join(f"{s} {pooled[s]:.1f}" for s in COL) + "   |   점프 에피소드 제외  " +
                        "  ".join(f"{s} {pooled[s + '_nojump']:.1f}" for s in COL) + "  [dB]", fontsize=9.5, loc="left")
        ax[j].set_xlabel("위성 고각 [deg]"); ax[j].set_ylabel("에피소드 NMSE [dB] (낮을수록 좋음)"); ax[j].grid(alpha=0.3)
        print(f"k={k}: " + ", ".join(f"{s} {v:.2f}" for s, v in pooled.items()), flush=True)
    handles = [Line2D([], [], color=COL[s], marker="o", ls="", label=s + (" (학습에 사용)" if s == "train" else "")) for s in COL]
    handles += [Line2D([], [], color="k", marker="o", ls="", label="LOS"), Line2D([], [], color="k", marker="x", ls="", label="NLOS"),
                Line2D([], [], color="k", marker="o", mfc="none", ms=11, ls="", label="타이밍 점프 에피소드")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=10)
    fig.suptitle("일반화 확인: 학습에 쓰지 않은 test·guard·val 에피소드가 같은 고각대의 train 에피소드와 같은 높이에 있는가",
                 fontsize=12, y=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(a.out, dpi=105)
    print("saved")


if __name__ == "__main__":
    main()
