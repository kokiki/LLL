"""
plot_ppt_data.py -- 디펜스 발표용 CSI 데이터 구성 그림 (16:9, 큰 글씨)
  figures/ppt_data_splits.png : split별 대표 에피소드 CSI 시계열 (train | val | test | guard), 에피소드·윈도우 수, 고각 범위
  figures/ppt_data_window.png : 에피소드 하나 위의 윈도우 구성 (history 20 ms -> k ms 뒤 예측, 1 ms씩 이동)
데이터: v9b, subcarrier 0, 안테나 (0,0), Re{h}를 에피소드별 최대 |h|로 정규화
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from data import Split

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
SPLITS = [("train", "C0", "학습"), ("val", "C2", "검증"), ("test", "C3", "테스트"), ("guard", "C1", "완충 구간")]
TP, TF = 20, 16
SNR_DB = 10.0      # 그림용 관측 SNR (학습은 5~20 dB 무작위)
ELEV_TEXT = {"train": "10–90° 중 나머지 구간", "val": "30–35°, 50–55°, 70–75°",
             "test": "12.5–14.5°, 20.5–24.5°, 40.5–44.5°,\n60.5–64.5°, 81–83°", "guard": "test 구간 양옆 1.5–2.5°\n(학습 제외 → train과 test를 1.8° 이상 분리)"}


def pick(sp):
    el = sp.meta["elevDeg"].astype(float)
    cand = np.where(~sp.nlos)[0]
    e = int(cand[np.argmin(np.abs(el[cand] - np.median(el[cand])))])
    h = sp.X[e, 0, :, 0].cpu().numpy()
    return e, el[e], h / np.abs(h).max()                 # 정규화 복소 CSI


def fig_splits(sps):
    fig, ax = plt.subplots(figsize=(16, 7.2))
    T = 150
    for i, (s, c, kor) in enumerate(SPLITS):
        sp = sps[s]; e, el, y = pick(sp); y = y.real
        x0 = i * (T + 12)                                   # 구간 사이 간격 (서로 다른 에피소드임을 표시)
        ax.axvspan(x0, x0 + T, color=c, alpha=0.10, lw=0)
        ax.plot(x0 + np.arange(T), y, color=c, lw=2.4)
        nw = sp.E * (T - TP - TF + 1)
        ax.text(x0 + T / 2, 1.62, f"{s}  ({kor})", ha="center", va="center", fontsize=19, fontweight="bold", color=c)
        ax.text(x0 + T / 2, 1.30, f"에피소드 {sp.E}개 · 윈도우 {nw:,}개", ha="center", va="center", fontsize=14)
        ax.text(x0 + T / 2, -1.42, f"고각 {ELEV_TEXT[s]}", ha="center", va="top", fontsize=12.5, color="0.25")
        ax.text(x0 + 3, -1.12, f"예: 고각 {el:.1f}°, LOS", fontsize=11, color="0.35")
        ax.plot([x0, x0 + T], [-1.22, -1.22], color=c, lw=1.2)
        ax.text(x0 + T, -1.17, "150 ms", ha="right", fontsize=11, color="0.35")
    ax.set_xlim(-5, 4 * (T + 12) - 7); ax.set_ylim(-1.95, 1.85)
    ax.set_yticks([-1, 0, 1]); ax.tick_params(axis="y", labelsize=12); ax.set_xticks([])
    ax.set_ylabel("정규화 CSI  Re{h}", fontsize=15)
    for side in ("top", "right", "bottom"): ax.spines[side].set_visible(False)
    ax.set_title("CSI 데이터 구성: 고각 구간 단위로 에피소드(150 ms)를 train / val / test / guard에 배정",
                 fontsize=20, pad=16)
    fig.text(0.5, 0.015, "총 400개 에피소드 · 1 ms 간격 관측 · 16x16 위성 배열 x subcarrier 12개 · "
             "각 에피소드는 한 split에만 속함 (구간 사이 끊김 = 서로 다른 에피소드)", ha="center", fontsize=12.5, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("figures/ppt_data_splits.png", dpi=200)


def fig_window(sps):
    sp = sps["train"]; e, el, hc = pick(sp)
    T, k, w0 = 150, 8, 50
    y = hc.real
    # 학습과 같은 노이즈: SNR 10 dB, 노이즈 전력 = history 구간 신호 전력 x 10^(-SNR/10) (복소 가우시안)
    rng = np.random.default_rng(3)
    hist = hc[w0:w0 + TP]
    sig = np.mean(np.abs(hist) ** 2)
    noisy = hist + np.sqrt(sig * 10 ** (-SNR_DB / 10) / 2) * (rng.standard_normal(TP) + 1j * rng.standard_normal(TP))
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.plot(t, y, color="0.55", lw=2.2, zorder=1)
    h0, h1, tgt = w0, w0 + TP - 1, w0 + TP - 1 + k
    ax.axvspan(h0 - 0.5, h1 + 0.5, color="C0", alpha=0.18, lw=0)
    ax.plot(t[h0:h1 + 1], y[h0:h1 + 1], "--", color="C0", lw=1.6, alpha=0.7, zorder=2, label="실제 채널 (노이즈 없음)")
    ax.plot(t[h0:h1 + 1], noisy.real, "o-", color="C0", lw=2.2, ms=7, zorder=3, label=f"관측값 (노이즈 포함, SNR {SNR_DB:g} dB)")
    ax.legend(loc="lower left", fontsize=13, framealpha=0.95)
    ax.axvspan(h1 + 0.5, h1 + TF + 0.5, color="C1", alpha=0.08, lw=0)
    ax.plot(tgt, y[tgt], "*", color="C3", ms=26, mec="k", zorder=4)
    ax.annotate("", xy=(tgt, y[tgt]), xytext=(h1, noisy.real[-1]),
                arrowprops=dict(arrowstyle="-|>", color="C3", lw=2.4, connectionstyle="arc3,rad=-0.35"))
    yt = 1.66
    ax.text(h0 + 1, yt, f"입력: history {TP} ms\n(노이즈 포함 관측 {TP}개)", ha="left", va="center", fontsize=16, color="C0", fontweight="bold")
    ax.text(h1 + TF + 2, 1.45, f"예측 대상: k ms 뒤 CSI (★, 그림은 k = {k})", ha="left", va="center", fontsize=16, color="C3",
            fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ax.text(h1 + TF / 2 + 0.5, -1.38, f"미래 구간 {TF} ms\n(k = 1–16 ms)", ha="center", va="center", fontsize=13, color="C1")
    arr = FancyArrowPatch((h0, -1.72), (h0 + 18, -1.72), arrowstyle="-|>", mutation_scale=22, color="0.3", lw=2)
    ax.add_patch(arr)
    ax.text(h0 + 20, -1.72, "1 ms씩 이동하며 윈도우 생성 → 에피소드당 115개 (= 150 - 20 - 16 + 1)",
            va="center", fontsize=14, color="0.25")
    ax.set_xlim(0, T); ax.set_ylim(-1.95, 2.0)
    ax.set_xlabel("시간 [ms]  (1 ms 간격 관측, 에피소드 하나 = 150 ms)", fontsize=15)
    ax.set_ylabel("정규화 CSI  Re{h}", fontsize=15)
    ax.tick_params(labelsize=12)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.set_title(f"학습 샘플(윈도우) 구성: 과거 {TP} ms 관측으로 k ms 뒤 CSI를 예측", fontsize=20, pad=16)
    fig.text(0.5, 0.015, f"윈도우 하나 x subcarrier 12개 = 모델 샘플 12개 (입력 {TP} x 16x16 복소 CSI → 출력 16x16 복소 CSI)  ·  "
             f"학습 시 관측 SNR 5–20 dB 무작위, 예측 정답은 노이즈 없는 실제 채널  ·  예: train 에피소드, 고각 {el:.1f}°, LOS",
             ha="center", fontsize=12, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("figures/ppt_data_window.png", dpi=200)


if __name__ == "__main__":
    sps = {s: Split(D, s, device="cpu") for s, _, _ in SPLITS}
    fig_splits(sps)
    fig_window(sps)
    print("saved")
