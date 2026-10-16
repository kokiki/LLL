"""
eval_array_impair_v10.py -- 위성 배열 비이상성에서 안테나 영역 딥러닝 vs 배열 구조(매니폴드) 기반 고전 기법
  관측 채널 = 정답 채널 = c ⊙ h   (c: 소자별 고정 복소 응답, 배열 1개 -> 모든 에피소드·시간·subcarrier 공통)
    위상·이득 오차: φ ~ N(0, σφ²), 이득 ~ 10^(N(0, σg²)/20)    (compare_models.py robust와 같은 모델)
    고장 소자    : 무작위 p% 소자 응답 0
  프리코딩에 필요한 것은 하드웨어를 포함한 실효 채널이므로 정답도 c ⊙ h. 노이즈는 로더가 c ⊙ h에 주입(SNR 5-20 dB).
  모델·baseline은 이상적 배열(train/val)로 학습·적합한 그대로 (오차 데이터 학습 없음). 오차 실현값 n_draw개 평균.
  출력: 시나리오 x k별 NMSE (전체·LOS·NLOS), BSC-UNet-AI - ManifoldPoly2 에피소드 bootstrap
        -> runs/eval_array_impair_v10_{split}.json
"""
import argparse
import json
import os

import numpy as np
import torch

from baselines import Baselines
from compare_models import load
from data import Split, Windows, from_model_out, to_model_io

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10"
Tp, Tf = 20, 16
C128 = torch.complex128
BASES = ["Poly2_blind", "ManifoldLinReg", "ManifoldPoly2"]
SCEN = [("이상적 배열", 0.0, 0.0, 0.0),
        ("위상 5° / 이득 0.5 dB", 5.0, 0.5, 0.0), ("위상 10° / 이득 1 dB", 10.0, 1.0, 0.0), ("위상 20° / 이득 2 dB", 20.0, 2.0, 0.0),
        ("고장 소자 5%", 0.0, 0.0, 0.05), ("고장 소자 10%", 0.0, 0.0, 0.10)]


def make_c(ph_deg, g_db, p_fail, seed):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(256, generator=g) * np.deg2rad(ph_deg)
    amp = 10 ** (torch.randn(256, generator=g) * g_db / 20)
    c = amp * torch.exp(1j * phi)
    if p_fail > 0:
        c[torch.randperm(256, generator=g)[:int(round(p_fail * 256))]] = 0
    return c.to(torch.complex64)


@torch.no_grad()
def run(W, E, ks, models, bl):
    names = list(models) + BASES
    num = {m: np.zeros((E, len(ks))) for m in names}; den = np.zeros((E, len(ks)))
    for d in W.eval_batches(16):
        e = d["e"].cpu().numpy(); B, S = d["hist_n"].shape[:2]
        bp = bl.predict(d)
        for j, k in enumerate(ks):
            Y = d["fut"][:, :, k - 1].to(C128)
            np.add.at(den[:, j], e, (Y.abs() ** 2).sum(dim=(1, 2)).cpu().numpy())
            for m, Gk in models.items():
                p = from_model_out(Gk[k](to_model_io(d, k)[0]), B, S).to(C128)
                np.add.at(num[m][:, j], e, ((p - Y).abs() ** 2).sum(dim=(1, 2)).cpu().numpy())
            for m in BASES:
                np.add.at(num[m][:, j], e, ((bp[m][:, :, j] - Y).abs() ** 2).sum(dim=(1, 2)).cpu().numpy())
    return num, den


def boot(nA, nB, den, n=10000, seed=0):
    """에피소드 bootstrap: Δ = NMSE_A - NMSE_B [dB] (pooled), 95% CI, A 우세 에피소드 수."""
    rng = np.random.default_rng(seed); E = len(den)
    idx = rng.integers(0, E, (n, E))
    d = 10 * np.log10(nA[idx].sum(1) / den[idx].sum(1)) - 10 * np.log10(nB[idx].sum(1) / den[idx].sum(1))
    full = 10 * np.log10(nA.sum() / den.sum()) - 10 * np.log10(nB.sum() / den.sum())
    return full, np.percentile(d, 2.5), np.percentile(d, 97.5), int((nA < nB).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "guard"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--n_draw", type=int, default=3)
    a = ap.parse_args()
    tr, va, ev = Split(D, "train"), Split(D, "val"), Split(D, a.split)
    bl = Baselines(Tp, a.ks, 1.0).fit(Windows(tr, Tp, Tf), Windows(va, Tp, Tf))
    models = {"BSC-UNet-AI": {k: load(f"runs/v10_Tp20_bscAI_k{k}ms/best_G.pt", Tp, "cuda") for k in a.ks},
              "BSC-UNet-I": {k: load(f"runs/v10_Tp20_bscI_k{k}ms/best_G.pt", Tp, "cuda") for k in a.ks}}
    if all(os.path.exists(f"runs/v10_Tp20_bscAIR_k{k}ms/best_G.pt") for k in a.ks):     # 배열 비이상성 증강 학습 모델
        models["BSC-UNet-AIR"] = {k: load(f"runs/v10_Tp20_bscAIR_k{k}ms/best_G.pt", Tp, "cuda") for k in a.ks}
    names = list(models) + BASES
    X0 = ev.X.clone(); los = ~ev.nlos.cpu().numpy() if torch.is_tensor(ev.nlos) else ~np.asarray(ev.nlos)
    out = {}
    for si, (lab, ph, gdb, pf) in enumerate(SCEN):
        draws = 1 if (ph == 0 and gdb == 0 and pf == 0) else a.n_draw
        N = {m: 0 for m in names}; Dn = 0
        for r in range(draws):
            ev.X = X0 * make_c(ph, gdb, pf, 1000 + 17 * si + r).to(X0.device)
            num, den = run(Windows(ev, Tp, Tf), ev.E, a.ks, models, bl)
            N = {m: N[m] + num[m] for m in names}; Dn = Dn + den
        res = {}
        for j, k in enumerate(a.ks):
            r = {}
            for g, mk in (("all", np.ones(ev.E, bool)), ("LOS", los), ("NLOS", ~los)):
                r[g] = {m: 10 * np.log10(N[m][mk, j].sum() / Dn[mk, j].sum()) for m in names}
            f, lo, hi, w = boot(N["BSC-UNet-AI"][:, j], N["ManifoldPoly2"][:, j], Dn[:, j])
            r["AI_minus_MP2"] = {"delta": f, "ci": [lo, hi], "wins": w, "E": int(ev.E)}
            if "BSC-UNet-AIR" in models:
                for ref in ("ManifoldPoly2", "Poly2_blind"):
                    f, lo, hi, w = boot(N["BSC-UNet-AIR"][:, j], N[ref][:, j], Dn[:, j])
                    r[f"AIR_minus_{ref}"] = {"delta": f, "ci": [lo, hi], "wins": w, "E": int(ev.E)}
            res[str(k)] = r
        out[lab] = res
        print(f"\n=== {lab} ({draws} draw) ===", flush=True)
        print(f"{'k':>4s} {'group':5s}" + "".join(f"{m:>16s}" for m in names))
        for k in a.ks:
            for g in ("all", "LOS", "NLOS"):
                print(f"{k:4d} {g:5s}" + "".join(f"{res[str(k)][g][m]:16.2f}" for m in names))
            s = res[str(k)]["AI_minus_MP2"]
            print(f"     AI - ManifoldPoly2 = {s['delta']:+.2f} dB CI[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] "
                  f"{'*' if s['ci'][1] < 0 or s['ci'][0] > 0 else ' '} AI 우세 {s['wins']}/{s['E']}")
            for ref in ("ManifoldPoly2", "Poly2_blind"):
                if f"AIR_minus_{ref}" in res[str(k)]:
                    s = res[str(k)][f"AIR_minus_{ref}"]
                    print(f"     AIR - {ref} = {s['delta']:+.2f} dB CI[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] "
                          f"{'*' if s['ci'][1] < 0 or s['ci'][0] > 0 else ' '} AIR 우세 {s['wins']}/{s['E']}")
    ev.X = X0
    json.dump(out, open(f"runs/eval_array_impair_v10_{a.split}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2,
              default=float)
    print(f"\nSaved runs/eval_array_impair_v10_{a.split}.json")


if __name__ == "__main__":
    main()
