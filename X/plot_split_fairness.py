"""
plot_split_fairness.py -- train/val/test/guard 분할의 공정성·신뢰성 확인 그림 (2x2)
  (a) val·test·guard 각 에피소드에서 가장 가까운 train 에피소드까지 고각 거리 분포 (test는 학습 고각과 겹치지 않음)
  (b) split별 채널 조건: 지연 확산, LOS K-factor 분포와 NLOS 비율 (split 간 채널 조건이 공정하게 비슷한지)
  (c) 채널 실현 독립성: 각 에피소드와 가장 비슷한 train 에피소드의 산란 성분(주 공간 성분 제거) 상관
      (같은 실현이 split을 넘어 중복되면 1에 가까움)
  (d) 누수 점검 T3: 학습한 k의 모델 출력을 모든 미래 시점 k'의 정답과 비교 -> 최소가 학습한 k여야 정상
출력: figures/split_fairness_v9b.png
"""
import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
EVAL = "./runs/eval_v9b_Tp20_test.json"
COL = {"train": "C0", "val": "C2", "test": "C3", "guard": "C1"}


def resid_unit(X):
    """에피소드별 주 공간 성분(rank-1) 제거한 산란 성분 스트림 -> 단위 벡터 (E, n)."""
    out = []
    for x in X:
        M = x.reshape(-1, 256).to(torch.complex128)
        _, _, Vh = torch.linalg.svd(M, full_matrices=False)
        v = Vh[0].conj()
        out.append((M - (M @ v)[:, None] * v.conj()[None]).flatten())
    R = torch.stack(out)
    return R / torch.linalg.vector_norm(R, dim=1, keepdim=True)


def main():
    sp = {s: Split(D, s) for s in COL}
    el = {s: v.meta["elevDeg"].astype(float) for s, v in sp.items()}
    fig, ax = plt.subplots(2, 2, figsize=(16, 10.5))

    # (a) 가장 가까운 train 고각까지 거리
    a = ax[0, 0]
    bins = np.linspace(0, 4, 33)
    d_tr = np.abs(el["train"][:, None] - el["train"][None]); np.fill_diagonal(d_tr, np.inf)
    a.hist(d_tr.min(1), bins=bins, color=COL["train"], alpha=0.35, label=f"train ↔ 다른 train (참고)")
    for s in ("val", "guard", "test"):
        d = np.abs(el[s][:, None] - el["train"][None]).min(1)
        a.hist(d, bins=bins, histtype="step", lw=2.2, color=COL[s], label=f"{s} ↔ train (최소 {d.min():.2f}°)")
    a.set_xlabel("가장 가까운 train 에피소드와의 고각 차이 [deg]"); a.set_ylabel("에피소드 수")
    a.set_title("(a) test는 학습 고각과 최소 1.8° 이상 떨어짐 (guard·val은 train에 인접)", fontsize=10.5, loc="left")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # (b) 채널 조건
    a = ax[0, 1]
    names = list(COL)
    ds = [sp[s].meta["delaySpread"].astype(float) * 1e9 for s in names]
    pos = np.arange(len(names))
    bp = a.boxplot(ds, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
    for p, s in zip(bp["boxes"], names): p.set_facecolor(COL[s]); p.set_alpha(0.5)
    a.set_yscale("log"); a.set_ylabel("지연 확산 [ns] (상자, 왼쪽 축)")
    a2 = a.twinx()
    kd = [sp[s].meta["KdB"].astype(float)[~sp[s].nlos] for s in names]
    bp2 = a2.boxplot([k[np.isfinite(k)] for k in kd], positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
    for p, s in zip(bp2["boxes"], names): p.set_facecolor("white"); p.set_edgecolor(COL[s]); p.set_hatch("//")
    a2.set_ylabel("LOS K-factor [dB] (빗금 상자, 오른쪽 축)")
    a.set_xticks(pos)
    a.set_xticklabels([f"{s}\nE={sp[s].E}, NLOS {100 * sp[s].nlos.mean():.0f}%" for s in names])
    a.set_title("(b) split별 채널 조건 (지연 확산·K-factor·NLOS 비율)", fontsize=10.5, loc="left"); a.grid(alpha=0.3, axis="y")

    # (c) 채널 실현 독립성
    a = ax[1, 0]
    Rtr = resid_unit(sp["train"].X)
    C = (Rtr.conj() @ Rtr.T).abs(); C.fill_diagonal_(0)
    bins = np.linspace(0, 0.05, 41)
    a.hist(C.max(1).values.cpu().numpy(), bins=bins, color=COL["train"], alpha=0.35, label="train ↔ 다른 train (참고)")
    for s in ("val", "guard", "test"):
        R = resid_unit(sp[s].X)
        m = (R.conj() @ Rtr.T).abs().max(1).values.cpu().numpy()
        a.hist(m, bins=bins, histtype="step", lw=2.2, color=COL[s], label=f"{s} ↔ train (최대 {m.max():.3f})")
    a.set_xlabel("가장 비슷한 train 에피소드와의 산란 성분 상관 (1 = 같은 채널 실현)"); a.set_ylabel("에피소드 수")
    a.set_title("(c) 채널 실현 중복 없음: 모든 상관이 0.05 미만 (독립 실현)", fontsize=10.5, loc="left")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # (d) 누수 점검 T3
    a = ax[1, 1]
    r = json.load(open(EVAL))["leak_check"]
    for i, (n, v) in enumerate(sorted(r.items(), key=lambda kv: kv[1]["trained_k"])):
        c = np.array(v["T3_output_vs_horizon_nmse_db"]); k = v["trained_k"]; kk = np.arange(1, len(c) + 1)
        line, = a.plot(kk, c, lw=2, marker="o", ms=3, label=f"k={k} 모델 (T2 {v['T2_shuffled_target_nmse_db']:+.1f} dB)")
        a.plot(k, c[k - 1], marker="*", ms=16, color=line.get_color(), mec="k")
    a.set_xlabel("비교한 정답 시점 k' [ms]"); a.set_ylabel("NMSE [dB]")
    a.set_title("(d) 누수 점검: 각 모델의 오차가 학습한 k(★)에서 최소, 다른 에피소드 정답과는 +3 dB(T2)", fontsize=10.5, loc="left")
    a.legend(fontsize=8.5, ncol=2); a.grid(alpha=0.3)

    fig.suptitle("v9b 데이터 분할의 공정성·신뢰성 확인 (고각 구간 block split, 에피소드 단위 배정)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("figures/split_fairness_v9b.png", dpi=105)
    print("saved")


if __name__ == "__main__":
    main()
