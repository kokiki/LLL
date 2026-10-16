"""
plot_ppt_slideA.py -- 디펜스 슬라이드 A (데이터 구성) 그림 2장, 16:9
  figures/ppt_A1_episode.png : 에피소드 150 ms 타임라인 — 에피소드마다 고정되는 값(점선) vs 에피소드 안에서 연속 변화하는 값(곡선)
  figures/ppt_A2_window.png  : 윈도우 구조 — history 20 ms + horizon k 화살표, subcarrier stride 25, Python 로더 온더플라이 처리
실제 v9b 에피소드 메타데이터·CSI를 사용 (LO 위상잡음은 생성기와 같은 통계 0.3°/√ms 랜덤워크로 예시)
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from data import Split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
T = 150
FIX_C, VAR_C = "#5a6b7d", "#c0392b"


def lane(ax, y, x, v, color, lw=2.6, ls="-"):
    v = np.asarray(v, float)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    ax.plot(x, y - 0.32 + 0.64 * v, color=color, lw=lw, ls=ls)


def fig_episode():
    sp = Split(D, "test", device="cpu")
    el = sp.meta["elevDeg"].astype(float)
    cand = np.where(~sp.nlos)[0]; e = int(cand[np.argmin(np.abs(el[cand] - np.median(el[cand])))])
    m = {k: float(v[e]) for k, v in sp.meta.items() if np.issubdtype(np.asarray(v).dtype, np.number)}
    h = sp.X[e, 0, :, 0].cpu().numpy()
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(16, 9))
    fixed = [("LOS / NLOS 상태", "LOS"),
             ("위성 고각", f"{m['elevDeg']:.1f}°"),
             ("K-factor", f"{m['KdB']:.1f} dB"),
             ("지연 확산 (delay spread)", f"{m['delaySpread'] * 1e9:.1f} ns"),
             ("shadow fading", f"{m['shadowFadeDB']:.1f} dB"),
             ("배열 방위각 (bearing)", "0° (전체 고정)"),
             ("궤도 Doppler 사전보상량", f"{m['fD_sat'] / 1e3:.0f} kHz")]
    rng = np.random.default_rng(1)
    pn = np.cumsum(0.3 * rng.standard_normal(T))                                    # deg, 0.3°/√ms 랜덤워크
    ph2 = np.pi * m["fDdot_sat"] * (t * 1e-3) ** 2                                  # Doppler 변화율 잔차 2차 위상 [rad]
    varying = [("Doppler 변화율 잔차 위상 (2차)", ph2, f"π·(dfD/dt)·t², dfD/dt = {m['fDdot_sat']:.0f} Hz/s"),
               ("LO 위상잡음 (랜덤워크)", pn, "σ = 0.3°/√ms"),
               ("소규모 페이딩 |h(t)|", np.abs(h), "다중경로 + 단말 움직임\n(UE Doppler 약 4.6 Hz)"),
               ("CSI Re{h(t)} (합성 결과)", h.real, "모델이 예측하는 신호")]
    n_f, n_v = len(fixed), len(varying)
    y = n_f + n_v + 1.2
    ax.text(-2, y, "에피소드마다 1회 샘플링 후 150 ms 동안 고정", fontsize=17, fontweight="bold", color=FIX_C, ha="left")
    for name, val in fixed:
        y -= 1
        ax.plot([0, T], [y, y], ls=(0, (6, 4)), color=FIX_C, lw=2.2)
        ax.text(-3, y, name, ha="right", va="center", fontsize=14)
        ax.text(T + 3, y, val, ha="left", va="center", fontsize=14, color=FIX_C, fontweight="bold")
    y -= 1.25
    ax.text(-2, y + 0.45, "에피소드 안에서 연속 변화", fontsize=17, fontweight="bold", color=VAR_C, ha="left")
    for name, v, note in varying:
        y -= 1
        lane(ax, y, t, v, VAR_C if "Re{h" not in name else "C0")
        ax.text(-3, y, name, ha="right", va="center", fontsize=14)
        ax.text(T + 3, y, note, ha="left", va="center", fontsize=12, color="0.3")
    ax.axhline(y + n_v + 0.55, xmin=0.0, xmax=1.0, color="0.85", lw=1)
    ax.set_xlim(-2, T + 2); ax.set_ylim(y - 0.8, n_f + n_v + 1.8)
    ax.set_yticks([]); ax.set_xticks(np.arange(0, T + 1, 25)); ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("시간 [ms]   (에피소드 1개 = 150 ms = 1 ms 간격 관측 150개)", fontsize=15)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title(f"에피소드 구조: 고정값(점선) vs 연속 변화값(곡선) — 예: test 에피소드, 고각 {m['elevDeg']:.1f}°, LOS",
                 fontsize=19, pad=14)
    fig.text(0.5, 0.012, "곡선은 각 성분의 시간 변화 모양만 보이도록 정규화 · 고각은 에피소드 사이(위성 통과 궤적을 따라 400개 배치)에서만 변함",
             ha="center", fontsize=12.5, color="0.35")
    fig.subplots_adjust(left=0.2, right=0.78, top=0.92, bottom=0.1)
    fig.savefig("figures/ppt_A1_episode.png", dpi=200)


def fig_window():
    TP, w0 = 20, 40
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.06, 0.40, 0.88, 0.50])
    ax.add_patch(Rectangle((0, 0), T, 1, facecolor="0.93", edgecolor="0.5", lw=1.5))
    ax.text(T / 2, 1.18, "에피소드 1개 = 150 ms (1 ms 간격 관측 150개)", ha="center", fontsize=16)
    ax.add_patch(Rectangle((w0, 0), TP, 1, facecolor="C0", alpha=0.75))
    ax.text(w0 + TP / 2, 0.5, f"history\n{TP} ms", ha="center", va="center", fontsize=16, color="white", fontweight="bold")
    t0 = w0 + TP
    ks = [1, 2, 4, 8, 12, 16, 24, 32]
    for i, k in enumerate(ks):
        yk = -0.35 - 0.33 * i
        long = k > 16
        ax.annotate("", xy=(t0 + k, yk), xytext=(t0, yk),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color="C3" if not long else "C4", ls="-" if not long else "--"))
        ax.plot([t0 + k, t0 + k], [yk, 0], color="C3" if not long else "C4", lw=0.8, ls=":")
        ax.text(t0 - 1.5, yk, f"k = {k} ms", va="center", ha="right", fontsize=13, color="C3" if not long else "C4")
    ax.text(t0 + 36, -0.9, "k = 1–16 ms: 미래 구간 Tf = 16\nk = 24, 32 ms: Tf = k (LEO 최대 왕복 지연 약 26 ms 대응)",
            fontsize=13, va="center", color="0.25")
    ax.annotate("", xy=(w0 + 12, 1.55), xytext=(w0, 1.55), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="0.3"))
    ax.text(w0 + 13, 1.55, "1 ms씩 이동 (win_stride = 1) → 에피소드당 115개 윈도우 (k ≤ 16 기준)", va="center", fontsize=13.5, color="0.25")
    ax.set_xlim(-2, T + 2); ax.set_ylim(-3.2, 1.8); ax.axis("off")
    # subcarrier stride
    ax2 = fig.add_axes([0.06, 0.24, 0.40, 0.09])
    sc = np.arange(300)
    ax2.vlines(sc, 0, 0.5, color="0.8", lw=0.6)
    ax2.vlines(sc[::25], 0, 1, color="C2", lw=2.4)
    ax2.set_xlim(-3, 302); ax2.set_ylim(0, 1.6); ax2.axis("off")
    ax2.text(150, 1.25, "subcarrier 300개 중 stride 25 → 12개 사용 (각각 독립 샘플)", ha="center", fontsize=13.5)
    # 온더플라이 처리
    ax3 = fig.add_axes([0.50, 0.06, 0.46, 0.28]); ax3.axis("off")
    ax3.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.94, boxstyle="round,pad=0.01", facecolor="#f4f6fa", edgecolor="0.6"))
    steps = ["Python 로더에서 온더플라이 처리 (MATLAB은 원본 CSI만 저장)",
             "① 윈도우 자르기 (history 20 ms + 미래 Tf)",
             "② 인과적 정규화 (history 마지막까지의 최대 크기)",
             "③ 채널 추정 오차 주입: SNR 5–20 dB 무작위 (정답은 노이즈 없음)",
             "④ Doppler 잔차 회전 제거 (noisy history에서 추정)"]
    for i, s in enumerate(steps):
        ax3.text(0.04, 0.84 - 0.19 * i, s, fontsize=14 if i == 0 else 13, fontweight="bold" if i == 0 else "normal",
                 transform=ax3.transAxes, va="center")
    fig.text(0.06, 0.12, "저장 용량 절감 + Tp, k, SNR을\n재생성 없이 바꿔 실험 가능", fontsize=13.5, color="0.3")
    fig.suptitle("윈도우 구조: 과거 20 ms 관측으로 k ms 뒤 CSI 예측", fontsize=20, y=0.975)
    fig.savefig("figures/ppt_A2_window.png", dpi=200)


if __name__ == "__main__":
    fig_episode()
    fig_window()
    print("saved")
