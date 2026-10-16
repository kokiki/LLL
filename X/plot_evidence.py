"""
plot_evidence.py -- 논문용 근거 그림 (v9b, 제안 모델 vs Poly2)
  ① 데이터 사용·분할 적정성
     (a) 고각 구간 block split + 실제 에피소드(LOS/NLOS), split별 에피소드·윈도우 수
     (b) 에피소드별 NMSE vs 고각 (학습 미사용 test·guard·val이 같은 고각대 train과 같은 높이인가)
  ② 예측 CSI가 실제 미래 CSI와 유사한가 (배열 전체 기준)
     (c) 빔 계수 파형: 에피소드의 주 공간 방향 v (clean 스트림 주 고유벡터)로 256 안테나를 묶은 c(t) = v^H h(t)
         실제 / 제안 / Poly2, test LOS·NLOS 에피소드, k ms 앞 예측 (원래 스케일로 복원)
     (d) test 전체 평균 안테나별 NMSE 지도 (16x16): 제안, Poly2, 차이
     (e) 산점도: test 전체 예측 vs 실제 (Re, Im 원소), R² = 1 - Σ|p-y|²/Σ|y-ȳ|²
출력: figures/evidence_<tag>.png
"""
import argparse

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from baselines import Baselines
from compare_models import load
from data import Split, Windows, from_model_out, to_model_io

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
COL = {"train": "C0", "val": "C2", "test": "C3", "guard": "C1"}
TEST_B = [(12.5, 14.5), (20.5, 24.5), (40.5, 44.5), (60.5, 64.5), (81, 83)]
GW = [1.5, 2.5, 2.5, 2.5, 1.5]
VAL_B = [(30, 35), (50, 55), (70, 75)]
Tp, Tf = 20, 16
C = torch.complex128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v9b_Tp20_bscI_k{K}ms/best_G.pt")
    ap.add_argument("--label", default="BSC-UNet-I")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="figures/evidence_bscI.png")
    a = ap.parse_args()
    k = a.k
    sp = {s: Split(D, s) for s in COL}
    te = sp["test"]
    G = load(a.ckpt.replace("{K}", str(k)), Tp, "cuda")
    bl = Baselines(Tp, [k], 1.0).fit(Windows(sp["train"], Tp, Tf), Windows(sp["val"], Tp, Tf))
    nw = te.T - Tp - Tf + 1

    fig = plt.figure(figsize=(20, 17))
    gs = fig.add_gridspec(4, 6, height_ratios=[0.9, 1.0, 1.0, 1.05], hspace=0.45, wspace=0.45)

    # ---------------- (a) 고각 구간 분할 + 에피소드 ----------------
    ax = fig.add_subplot(gs[0, :3])
    ax.axvspan(10, 90, color=COL["train"], alpha=0.10)
    for (lo, hi), g in zip(TEST_B, GW):
        ax.axvspan(lo - g, lo, color=COL["guard"], alpha=0.25); ax.axvspan(hi, hi + g, color=COL["guard"], alpha=0.25)
        ax.axvspan(lo, hi, color=COL["test"], alpha=0.30)
    for lo, hi in VAL_B: ax.axvspan(lo, hi, color=COL["val"], alpha=0.30)
    rng = np.random.default_rng(0); lines = []
    for s, v in sp.items():
        el = v.meta["elevDeg"].astype(float); y = np.where(v.nlos, 0.3, 0.7) + rng.uniform(-0.1, 0.1, v.E)
        ax.scatter(el[~v.nlos], y[~v.nlos], s=12, color=COL[s]); ax.scatter(el[v.nlos], y[v.nlos], s=16, marker="x", color=COL[s])
        lines.append(f"{s}: 에피소드 {v.E}개 (NLOS {int(v.nlos.sum())}), 윈도우 {v.E * nw:,}개")
    ax.set_yticks([0.3, 0.7]); ax.set_yticklabels(["NLOS", "LOS"]); ax.set_ylim(0, 1); ax.set_xlim(10, 90)
    ax.set_xlabel("위성 고각 [deg]")
    ax.set_title("(a) 고각 구간 block split: 색 띠 = 구간 배정, 점 = 실제 에피소드 (에피소드 단위로 한 split에만 배정)", fontsize=10, loc="left")
    ax.legend(handles=[Patch(color=COL[s], alpha=0.4, label=l) for s, l in zip(COL, lines)], fontsize=8, loc="lower right")

    # ---------------- (b) 에피소드별 NMSE vs 고각 ----------------
    ax = fig.add_subplot(gs[0, 3:])
    pooled = {}
    with torch.no_grad():
        for s, v in sp.items():
            W = Windows(v, Tp, Tf, stride=3 if s == "train" else 1)
            num = np.zeros(v.E); den = np.zeros(v.E)
            for d in W.eval_batches(16):
                S = d["hist_n"].shape[1]; x, y = to_model_io(d, k)
                e = d["e"].repeat_interleave(S).cpu().numpy()
                np.add.at(num, e, ((G(x) - y) ** 2).sum(dim=(1, 2, 3)).double().cpu().numpy())
                np.add.at(den, e, (y ** 2).sum(dim=(1, 2, 3)).double().cpu().numpy())
            el = v.meta["elevDeg"].astype(float); nm = 10 * np.log10(num / den)
            pooled[s] = 10 * np.log10(num.sum() / den.sum())
            for nl, mk in ((False, "o"), (True, "x")):
                m = v.nlos == nl
                ax.scatter(el[m], nm[m], s=12 if s == "train" else 22, marker=mk, color=COL[s], alpha=0.5 if s == "train" else 0.95,
                           zorder=1 if s == "train" else 3)
    ax.set_xlabel("위성 고각 [deg]"); ax.set_ylabel("에피소드 NMSE [dB]"); ax.grid(alpha=0.3)
    ax.set_title(f"(b) {a.label} k={k} ms 에피소드별 NMSE: 학습 미사용 split이 같은 고각대 train과 같은 높이\n"
                 + "  ".join(f"{s} {v:.1f}" for s, v in pooled.items()) + " dB (전체)", fontsize=10, loc="left")
    ax.legend(handles=[Line2D([], [], color=COL[s], marker="o", ls="", label=s) for s in COL], fontsize=8, ncol=4)

    # ---------------- 예측 (test, 선택 에피소드 / 전체) ----------------
    el_te = te.meta["elevDeg"].astype(float)

    def predict(rows, seed=0):
        g = torch.Generator(device="cuda"); g.manual_seed(seed)
        d = Wte.batch(rows, g); B, S = d["hist_n"].shape[:2]
        with torch.no_grad():
            p = from_model_out(G(to_model_io(d, k)[0]), B, S).to(C)
            q = bl.predict(d)["Poly2_blind"][:, :, 0]
        return d, p, q, d["fut"][:, :, k - 1].to(C)

    Wte = Windows(te, Tp, Tf)
    # (c) 빔 계수 파형
    for c, nl in enumerate((False, True)):
        cand = np.where(te.nlos == nl)[0]
        e = int(cand[np.argmin(np.abs(el_te[cand] - np.median(el_te[cand])))])
        rows = torch.arange(e * nw, (e + 1) * nw, device="cuda")
        d, p, q, y = predict(rows)
        back = (te.cm[e, d["w"] + Tp - 1] * torch.exp(1j * d["omega"] * k)).to(C)[:, None, None]   # 모델 영역 -> 원래 스케일
        X = te.X[e, 0].to(C)                                                                            # subcarrier 0 (T, 256)
        _, U = torch.linalg.eigh(X.T @ X.conj()); v = U[:, -1]                                           # 주 공간 방향
        t_true = (d["w"] + Tp - 1 + k).cpu().numpy()
        cy = (X @ v.conj()).cpu().numpy()
        cp = ((p * back)[:, 0] @ v.conj()).cpu().numpy(); cq = ((q * back)[:, 0] @ v.conj()).cpu().numpy()
        nmp = 10 * np.log10(((p - y).abs() ** 2).sum().item() / (y.abs() ** 2).sum().item())
        nmq = 10 * np.log10(((q - y).abs() ** 2).sum().item() / (y.abs() ** 2).sum().item())
        ax = fig.add_subplot(gs[1, 3 * c:3 * c + 3])
        ax.plot(np.arange(te.T), cy.real, color="k", lw=2.6, alpha=0.35, label="실제 미래 CSI")
        ax.plot(t_true, cq.real, color="C1", lw=1.2, ls="--", label=f"Poly2 (NMSE {nmq:.1f} dB)")
        ax.plot(t_true, cp.real, color="C3", lw=1.6, label=f"{a.label} (NMSE {nmp:.1f} dB)")
        ax.set_xlim(0, te.T); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("시간 [ms]  (각 예측은 k ms 전까지 20 ms noisy history만 사용)"); ax.set_ylabel("Re{빔 계수 c(t)}")
        ax.set_title(f"(c) test {te.names[e]} ({'NLOS' if nl else 'LOS'}, 고각 {el_te[e]:.1f}°) {k} ms 앞 예측 — "
                     f"256 안테나 전체를 주 빔 방향으로 묶은 빔 계수", fontsize=9.5, loc="left")

    # test 전체: 안테나별 오차 누적 (d) + 산점도 표본 (e)
    num_p = torch.zeros(256, dtype=torch.float64, device="cuda"); num_q = torch.zeros_like(num_p); den = torch.zeros_like(num_p)
    yy, pp, qq = [], [], []
    rs = torch.Generator().manual_seed(0)
    for start in range(0, len(Wte), 64):
        rows = torch.arange(start, min(start + 64, len(Wte)), device="cuda")
        _, p, q, y = predict(rows, seed=start)
        num_p += ((p - y).abs() ** 2).sum(dim=(0, 1)); num_q += ((q - y).abs() ** 2).sum(dim=(0, 1)); den += (y.abs() ** 2).sum(dim=(0, 1))
        idx = torch.randint(0, y.numel(), (400,), generator=rs).to("cuda")
        yy.append(y.flatten()[idx].cpu()); pp.append(p.flatten()[idx].cpu()); qq.append(q.flatten()[idx].cpu())
    yy, pp, qq = (torch.cat(z).numpy() for z in (yy, pp, qq))

    # (d) test 평균 안테나별 NMSE 지도
    mp = (10 * torch.log10(num_p / den)).reshape(16, 16).cpu().numpy()
    mq = (10 * torch.log10(num_q / den)).reshape(16, 16).cpu().numpy()
    edge = np.zeros((16, 16), bool); edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    pdb = lambda n, m: 10 * np.log10(n.cpu().numpy()[m.ravel()].sum() / den.cpu().numpy()[m.ravel()].sum())
    lo, hi = min(mp.min(), mq.min()), max(mp.max(), mq.max())
    for i, (M, t) in enumerate(((mp, f"{a.label}\n안쪽 {pdb(num_p, ~edge):.1f} / 가장자리 {pdb(num_p, edge):.1f} dB"),
                                (mq, f"Poly2\n안쪽 {pdb(num_q, ~edge):.1f} / 가장자리 {pdb(num_q, edge):.1f} dB"))):
        ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2])
        im = ax.imshow(M, cmap="viridis_r", vmin=lo, vmax=hi); ax.set_title(t, fontsize=9.5)
        ax.set_xlabel("안테나 열"); ax.set_ylabel("안테나 행"); fig.colorbar(im, ax=ax, fraction=0.046, label="NMSE [dB] (밝을수록 좋음)")
    ax = fig.add_subplot(gs[2, 4:6])
    dm = mp - mq; vm = np.abs(dm).max()
    im = ax.imshow(dm, cmap="RdBu", vmin=-vm, vmax=vm)
    ax.set_title(f"차이 ({a.label} − Poly2) [dB]\n파랑 = {a.label} 우세 (우세 안테나 {int((dm < 0).sum())}/256)", fontsize=9.5)
    ax.set_xlabel("안테나 열"); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.5, gs[2, 0].get_position(fig).y1 + 0.012, f"(d) test 전체({len(Wte):,}개 윈도우 x subcarrier 12개) 평균 안테나별 NMSE, 16x16 위성 배열, {k} ms 앞 예측",
             ha="center", fontsize=10.5)

    # (e) 산점도
    for c, (z, name, colr) in enumerate(((pp, a.label, "C3"), (qq, "Poly2", "C1"))):
        ax = fig.add_subplot(gs[3, 3 * c:3 * c + 3])
        r2 = 1 - (np.abs(z - yy) ** 2).sum() / (np.abs(yy - yy.mean()) ** 2).sum()
        ax.scatter(yy.real, z.real, s=1, alpha=0.15, color=colr, rasterized=True)
        ax.scatter(yy.imag, z.imag, s=1, alpha=0.15, color=colr, rasterized=True)
        lim = np.percentile(np.abs(np.concatenate([yy.real, yy.imag])), 99.5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("실제 미래 CSI (Re, Im 원소, 모델 정규화 영역)"); ax.set_ylabel("예측 CSI"); ax.grid(alpha=0.3)
        ax.set_title(f"(e) test 전체 {k} ms 앞 예측 vs 실제 — {name}: R² = {r2:.4f}  (점선 = 완벽한 예측)", fontsize=10, loc="left")
    fig.suptitle(f"① 데이터 사용·분할 적정성 (a, b)   ② 예측 CSI와 실제 미래 CSI의 유사성 (c, d, e) — v9b, {a.label}, k={k} ms", fontsize=14)
    fig.savefig(a.out, dpi=95, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
