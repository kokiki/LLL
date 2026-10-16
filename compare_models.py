"""
compare_models.py -- (1) 제안 모델 간 paired 비교 (BSC-UNet vs LSC-UNet), (2) 배열 보정 오차 견고성 시험
===============================================================================
평가 배치는 evaluate.py와 동일 (고정 시드 12345, 16 윈도우/배치).
(1) horizon별로 두 체크포인트의 에피소드 단위 paired bootstrap (ΔNMSE 95% CI, 승리 에피소드 수)
(2) 배열 보정 오차: 소자별 고정 복소 이득 c_i = (1+g_i) e^{jφ_i}, φ_i ~ N(0, σ_φ²), 20log10(1+g_i) ~ N(0, σ_g²)
    실제 배열처럼 모든 에피소드·시간에 같은 c (배열 1개)를 적용: 관측 채널 = c ⊙ h (history와 정답 모두)
    모델과 baseline은 이상적 배열 데이터로 학습한 그대로 (오차 데이터로 재학습하지 않음). 오차 실현 n_draw개 평균.

Usage:
  python compare_models.py --data_root <v9> --split test --pair BSC=runs/v9_Tp20_beam_k{K}ms/best_G.pt \
      --pair LSC=runs/v9_Tp20_k{K}ms/best_G.pt --ks 1 2 4 8 12 16 --robust --out runs/compare_bsc_lsc_test.json
"""
import argparse
import json

import numpy as np
import torch

from baselines import Baselines
from data import Split, Windows, from_model_out, to_model_io
from csi_models import build_model_from_state_dict

C128 = torch.complex128
ROBUST_LEVELS = [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0), (20.0, 2.0)]      # (위상 rms [deg], 이득 rms [dB])
ROBUST_BASES = ["LinReg_eig", "Poly2_blind", "ManifoldLinReg"]


def load(path, Tp, dev):
    sd = torch.load(path, map_location=dev, weights_only=True)
    G = build_model_from_state_dict(sd, Tp)
    G.load_state_dict(sd)
    return G.to(dev).eval()


def run_eval(Wev, ev, ks, models, bl=None, bases=()):
    """models: {name: {k: G}} -> per-episode num (E, nk), den (E, nk)."""
    E, nk = ev.E, len(ks)
    num = {m: np.zeros((E, nk)) for m in list(models) + list(bases)}
    den = np.zeros((E, nk))
    with torch.no_grad():
        for d in Wev.eval_batches(16):
            e = d["e"].cpu().numpy(); B, S = d["hist_n"].shape[:2]
            Y = d["fut"][:, :, [k - 1 for k in ks]].to(C128)
            for j in range(nk):
                np.add.at(den[:, j], e, (Y[:, :, j].abs() ** 2).sum(dim=(1, 2)).cpu().numpy())
            for name, Gk in models.items():
                for j, k in enumerate(ks):
                    x, _ = to_model_io(d, k)
                    p = from_model_out(Gk[k](x), B, S).to(C128)
                    np.add.at(num[name][:, j], e, ((p - Y[:, :, j]).abs() ** 2).sum(dim=(1, 2)).cpu().numpy())
            if bases:
                bp = bl.predict(d)
                for m in bases:
                    for j in range(nk):
                        np.add.at(num[m][:, j], e, ((bp[m][:, :, j] - Y[:, :, j]).abs() ** 2).sum(dim=(1, 2)).cpu().numpy())
    return num, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["test", "guard", "val"])
    ap.add_argument("--pair", action="append", required=True, help="NAME=PATH_TEMPLATE ({K} -> horizon), 정확히 2개")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    ap.add_argument("--Tp", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--n_draw", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    assert len(a.pair) == 2
    tr, va, ev = (Split(a.data_root, s, device=dev) for s in ("train", "val", a.split))
    Tp, ks = a.Tp, a.ks
    Tf = max(max(ks), tr.Tf_file)
    names = [p.split("=", 1)[0] for p in a.pair]
    models = {p.split("=", 1)[0]: {k: load(p.split("=", 1)[1].replace("{K}", str(k)), Tp, dev) for k in ks} for p in a.pair}
    Wev = Windows(ev, Tp, Tf)
    groups = {"all": np.ones(ev.E, bool), "LOS": ~ev.nlos, "NLOS": ev.nlos}
    db = lambda n, d: float(10 * np.log10(n.sum() / d.sum()))
    rng = np.random.default_rng(0)
    report = {"split": a.split, "ks": ks, "pair": a.pair}

    # ---------------- (1) paired ----------------
    num, den = run_eval(Wev, ev, ks, models)
    A, B = names
    print(f"\n===== (1) paired: Δ = NMSE({A}) - NMSE({B})  [dB], 에피소드 bootstrap {a.n_boot}x =====")
    report["paired"] = {}
    for j, k in enumerate(ks):
        for g, mk in groups.items():
            idx = np.where(mk)[0]
            sel = idx[rng.integers(0, len(idx), (a.n_boot, len(idx)))]
            dA, dB_ = db(num[A][idx, j], den[idx, j]), db(num[B][idx, j], den[idx, j])
            bs = 10 * np.log10(num[A][sel, j].sum(1) / den[sel, j].sum(1)) - 10 * np.log10(num[B][sel, j].sum(1) / den[sel, j].sum(1))
            lo, hi = np.percentile(bs, [2.5, 97.5]); wins = int((num[A][idx, j] < num[B][idx, j]).sum())
            report["paired"].setdefault(str(k), {})[g] = {A: dA, B: dB_, "delta_db": dA - dB_, "ci95": [lo, hi], "wins": wins, "n_ep": len(idx)}
            sig = "*" if hi < 0 else ("x" if lo > 0 else " ")
            print(f"  k={k:2d} {g:5s} {A}={dA:7.2f} {B}={dB_:7.2f}  Δ={dA-dB_:+5.2f} CI[{lo:+5.2f},{hi:+5.2f}] {sig} wins {wins}/{len(idx)}")

    # ---------------- (2) 배열 보정 오차 ----------------
    if a.robust:
        Wtr, Wva = Windows(tr, Tp, Tf), Windows(va, Tp, Tf)
        bl = Baselines(Tp, ks, tr.dt_ms, dev).fit(Wtr, Wva)                  # 이상적 배열 train으로 적합 (모델과 동일 조건)
        X0 = ev.X.clone()
        report["robust"] = {}
        print("\n===== (2) 배열 보정 오차 견고성: NMSE [dB] (오차 실현 %d개 pooled) =====" % a.n_draw)
        for ph_deg, g_db in ROBUST_LEVELS:
            nums = None; dens = None
            draws = 1 if ph_deg == 0 and g_db == 0 else a.n_draw
            for r in range(draws):
                gen = torch.Generator(device="cpu").manual_seed(1000 + r)
                phi = torch.randn(256, generator=gen) * np.deg2rad(ph_deg)
                amp = 10 ** (torch.randn(256, generator=gen) * g_db / 20)
                c = (amp * torch.exp(1j * phi)).to(torch.complex64).to(dev)
                ev.X = X0 * c                                                    # 모든 에피소드·시간·subcarrier에 같은 배열 오차
                n_, d_ = run_eval(Wev, ev, ks, models, bl, ROBUST_BASES)
                nums = n_ if nums is None else {m: nums[m] + n_[m] for m in nums}
                dens = d_ if dens is None else dens + d_
            ev.X = X0
            lvl = f"phase{ph_deg:g}deg_gain{g_db:g}dB"
            for j, k in enumerate(ks):
                for g, mk in groups.items():
                    report["robust"].setdefault(lvl, {}).setdefault(str(k), {})[g] = {m: db(nums[m][mk, j], dens[mk, j]) for m in nums}
        cols = names + ROBUST_BASES
        for k in ks:
            print(f"\n  k={k}  " + "".join(f"{m:>15s}" for m in cols))
            for ph_deg, g_db in ROBUST_LEVELS:
                lvl = f"phase{ph_deg:g}deg_gain{g_db:g}dB"
                for g in ("all", "LOS", "NLOS"):
                    r = report["robust"][lvl][str(k)][g]
                    print(f"  φ{ph_deg:>4g}° g{g_db:>3g}dB {g:4s} " + "".join(f"{r[m]:15.2f}" for m in cols))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2, default=float)
        print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
