"""
plot_prediction_quality.py -- 예측 CSI vs 실제 CSI 비교 그림 (v9b test, BSC-UNet)
  (1) 파형 비교: test 에피소드(LOS·NLOS) 전체에서 매 시점의 k ms 예측(각자 직전 20 ms noisy history만 사용)을
      실제 CSI와 겹쳐 그림 (BSC-UNet, 최고 baseline). 예측은 모델 영역(정규화·Doppler 잔차 제거)에서 원래 스케일로 복원.
  (2) 윈도우별 NMSE 누적분포 (test 전체): BSC-UNet vs baseline
  (3) 예측 CSI로 복조할 때의 유효 SINR (새 파일럿 없이 예측 채널로 프리코딩·복조, 파일럿 오버헤드 절감 상황):
      송신 빔 w = conj(ĥ)/||ĥ|| (예측 CSI로 MRT), 단말은 예측 유효 채널 ĝ = ĥᵀw 로 복조, 실제 유효 채널 g = hᵀw
      SINR = |g|² / (|g - ĝ|² + σ²)  (예측 오차가 잔여 간섭으로 남음 -> 방향뿐 아니라 진폭·위상 오차가 직접 반영)
      채널 크기 정규화(|h| = 1), σ² = -20 dB (완벽한 CSI SNR 20 dB). 참고: 다중 사용자 ZF는 단말별 rank-1 채널에서
      프리코더가 방향에만 의존해(단말별 복소 상수 상쇄) 방법 간 차이가 거의 없음 (모두 약 19.3~19.9 dB).
출력: figures/prediction_quality_v9b.png
"""
import argparse

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
CK = "runs/v9b_Tp20_bsc_k{K}ms/best_G.pt"
Tp, KS = 20, (4, 16)
BASES = {"Poly2_blind": "Poly2", "ManifoldLinReg": "ManifoldLinReg", "LinReg_eig": "LinReg"}
W_STRIDE, SNR_DB = 5, 20.0
C128 = torch.complex128


def main():
    global CK
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CK, help="{K} -> horizon")
    ap.add_argument("--label", default="BSC-UNet")
    ap.add_argument("--out", default="figures/prediction_quality_v9b.png")
    a = ap.parse_args()
    CK = a.ckpt
    tr, va, te = (Split(D, s) for s in ("train", "val", "test"))
    Tf = 16
    Wtr, Wva, Wte = (Windows(s, Tp, Tf) for s in (tr, va, te))
    bl = Baselines(Tp, list(KS), 1.0).fit(Wtr, Wva)
    G = {k: load(CK.replace("{K}", str(k)), Tp, "cuda") for k in KS}
    nw = te.T - Tp - Tf + 1
    el = te.meta["elevDeg"].astype(float)

    def predict(rows, gen_seed=0):
        g = torch.Generator(device="cuda"); g.manual_seed(gen_seed)
        d = Wte.batch(rows, g)
        B, S = d["hist_n"].shape[:2]
        out = {"Y": d["fut"][:, :, [k - 1 for k in KS]].to(C128)}               # (B,S,nk,256) 모델 영역 정답
        bp = bl.predict(d)
        for m in list(BASES) + ["Persistence"]: out[m] = bp[m]
        out["BSC"] = torch.stack([from_model_out(G[k](to_model_io(d, k)[0]), B, S).to(C128) for k in KS], 2)
        return d, out

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05])
    # ---------------- (1) 시점별 NMSE (안테나 256개 x subcarrier 12개 전체) ----------------
    tsty = {"BSC": dict(color="C3", lw=2.0), "Poly2_blind": dict(color="C1", lw=1.4, ls="--"),
            "ManifoldLinReg": dict(color="C2", lw=1.4, ls="-."), "LinReg_eig": dict(color="C0", lw=1.2, ls=":")}
    for r, nl in enumerate((False, True)):
        cand = np.where(te.nlos == nl)[0]
        e = int(cand[np.argmin(np.abs(el[cand] - np.median(el[cand])))])
        rows = torch.arange(e * nw, (e + 1) * nw, device="cuda")
        with torch.no_grad():
            d, out = predict(rows)
        w = d["w"].double()
        for c, k in enumerate(KS):
            j = KS.index(k)
            t_true = (w + Tp - 1 + k).cpu().numpy()
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            Y = out["Y"][:, :, j]
            for m in ["BSC", "Poly2_blind", "ManifoldLinReg", "LinReg_eig"]:
                err = ((out[m][:, :, j] - Y).abs() ** 2).sum(dim=(1, 2)); den = (Y.abs() ** 2).sum(dim=(1, 2))
                nm_t = (10 * torch.log10(err / den)).cpu().numpy()                    # 시점별 NMSE (전체 안테나·subcarrier)
                nm = 10 * np.log10(err.sum().item() / den.sum().item())                  # 에피소드 전체 NMSE
                ax.plot(t_true, nm_t, label=f"{a.label if m == 'BSC' else BASES[m]} (에피소드 NMSE {nm:.1f} dB)", **tsty[m])
            ax.set_xlim(0, te.T); ax.grid(alpha=0.3); ax.set_ylabel("시점별 NMSE [dB] (낮을수록 좋음)")
            ax.set_title(f"test {te.names[e]} ({'NLOS' if nl else 'LOS'}, 고각 {el[e]:.1f}°) — {k} ms 앞 예측, "
                         f"안테나 256개 x subcarrier 12개 전체", fontsize=9.5, loc="left")
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            if r == 1: ax.set_xlabel("예측 대상 시점 [ms]  (각 예측은 k ms 전까지 20 ms noisy history만 사용)")

    # ---------------- (2) NMSE CDF, (3) 빔포밍 이득 손실 (test 전체) ----------------
    nm_w = {m: {k: [] for k in KS} for m in ["BSC"] + list(BASES)}
    keep_m = ["Y", "BSC"] + list(BASES) + ["Persistence"]
    store = {m: torch.zeros(te.E, (nw + W_STRIDE - 1) // W_STRIDE, te.n_sc, len(KS), 256, dtype=torch.complex64) for m in keep_m}
    with torch.no_grad():
        for start in range(0, len(Wte), 64):
            rows = torch.arange(start, min(start + 64, len(Wte)), device="cuda")
            d, out = predict(rows, gen_seed=start)
            for j, k in enumerate(KS):
                Y = out["Y"][:, :, j]
                for m in nm_w:
                    P = out[m][:, :, j]
                    nm_w[m][k].append((10 * torch.log10(((P - Y).abs() ** 2).sum(dim=(1, 2)) / (Y.abs() ** 2).sum(dim=(1, 2)))).cpu().numpy())
            sel = (d["w"] % W_STRIDE == 0)
            if sel.any():
                e_s, w_s = d["e"][sel].cpu(), (d["w"][sel] // W_STRIDE).cpu()
                for m in keep_m:
                    store[m][e_s, w_s] = out[m][sel].to(torch.complex64).cpu()
    # ---- 예측 CSI 복조 유효 SINR (단일 단말, 윈도우 x subcarrier) ----
    sig2 = 10 ** (-SNR_DB / 10)
    sinr = {m: {k: None for k in KS} for m in ["Perfect"] + keep_m[1:]}
    for j, k in enumerate(KS):
        H = store["Y"][..., j, :].to("cuda").to(C128)                              # (E, nw', S, 256)
        valid = torch.linalg.vector_norm(H, dim=-1) > 0                             # 저장된 윈도우만
        sc = torch.linalg.vector_norm(H, dim=-1, keepdim=True).clamp_min(1e-30)
        H = H / sc
        for m in sinr:
            P = H if m == "Perfect" else store[m][..., j, :].to("cuda").to(C128) / sc
            w = P.conj() / torch.linalg.vector_norm(P, dim=-1, keepdim=True).clamp_min(1e-30)
            g = (H * w).sum(-1); gh = (P * w).sum(-1)
            v = 10 * torch.log10(g.abs() ** 2 / ((g - gh).abs() ** 2 + sig2))
            sinr[m][k] = v[valid].cpu().numpy()
    sty = {"BSC": dict(color="C3", lw=2.4), "Poly2_blind": dict(color="C1", lw=1.6, ls="--"),
           "ManifoldLinReg": dict(color="C2", lw=1.6, ls="-."), "LinReg_eig": dict(color="C0", lw=1.6, ls=":"),
           "Perfect": dict(color="k", lw=1.4), "Persistence": dict(color="0.55", lw=1.6, ls=(0, (1, 1)))}
    lab = {"BSC": a.label, **BASES, "Perfect": "완벽한 CSI (상한)", "Persistence": "오래된 CSI (예측 없음)"}
    for c, k in enumerate(KS):
        ax = fig.add_subplot(gs[2, c])
        for m in nm_w:
            v = np.sort(np.concatenate(nm_w[m][k])); ax.plot(v, np.linspace(0, 1, len(v)), label=f"{lab[m]} (중앙값 {np.median(v):.1f} dB)", **sty[m])
        ax.set_xlabel("윈도우별 NMSE [dB]"); ax.set_ylabel("누적 비율 (CDF)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title(f"k={k} ms 윈도우별 NMSE (test {len(Wte)}개, 왼쪽 = 좋음)", fontsize=9.5, loc="left")
        ax = fig.add_subplot(gs[2, 2 + c])
        for m in ["Perfect", "BSC", "ManifoldLinReg", "Poly2_blind", "LinReg_eig", "Persistence"]:
            v = sinr[m][k]; rate = np.log2(1 + 10 ** (v / 10)).mean()
            v = np.sort(v); ax.plot(v, np.linspace(0, 1, len(v)), label=f"{lab[m]} (중앙값 {np.median(v):.1f} dB, {rate:.2f} b/s/Hz)", **sty[m])
        ax.set_xlim(-5, 22); ax.set_xlabel("유효 SINR [dB] (예측 CSI로 프리코딩·복조)"); ax.set_ylabel("누적 비율 (CDF)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7.2, loc="upper left")
        ax.set_title(f"k={k} ms 예측 CSI 복조 유효 SINR (오른쪽 = 좋음)", fontsize=9.5, loc="left")
    fig.suptitle(f"예측 CSI vs 실제 CSI (v9b test, {a.label} vs baseline)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=100)
    print("saved")


if __name__ == "__main__":
    main()
