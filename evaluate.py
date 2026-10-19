"""
evaluate.py -- 제안 기법(및 ablation) 체크포인트 + baseline을 같은 GPU 평가 배치에서 한 번에 평가
===============================================================================
- 같은 실행 안에서 모든 방법이 동일 윈도우·동일 노이즈(고정 시드 배치)를 공유 -> 비교 공정성이 구조적으로 보장
- baseline 통계/학습은 train(+val로 α 선택)에서만. 평가는 --split (test 또는 guard=확인용 test)
- 그룹: 전체 / LOS / NLOS / 고각<50 / 고각>=50 / 정답 시점 블로키지(주 경로 감쇠>3dB) / 비블로키지
- RCG vs baseline: 에피소드 단위 paired bootstrap (ΔNMSE 95% CI, P(Δ>=0)) + RCG 승리 에피소드 수
- --leak_check: (T2) 정답을 다른 에피소드 윈도우 것으로 교체, (T3) 같은 출력을 모든 horizon 정답과 비교

Usage:
  python evaluate.py --data_root <v9 dataset> --split test --ckpt RCG_k16=runs/.../best_G.pt:16 [--ckpt ...]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from baselines import NAMES, Baselines
from data import Split, Windows, from_model_out, rescale_doppler_rate, to_model_io
from csi_models import build_model_from_state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["test", "guard", "val"])
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH:K_MS (여러 번 가능). 딥러닝 baseline(models_dl)은 NAME=dl:<model>:PATH:K_MS")
    ap.add_argument("--ks_ms", type=float, nargs="*", default=None, help="baseline horizon들 (기본: ckpt horizon)")
    ap.add_argument("--Tp", type=int, default=None)
    ap.add_argument("--sc_stride", type=int, default=25)
    ap.add_argument("--win_stride", type=int, default=1)
    ap.add_argument("--doppler", default="est", choices=["est", "est_parity", "none"], help="est_parity: v7 이전 데이터 전용")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--leak_check", action="store_true")
    ap.add_argument("--elev_range", type=float, nargs=2, default=None, help="모든 split을 이 고각 범위 에피소드로 제한 (비교 실험용)")
    ap.add_argument("--doppler_rate_kappa", type=float, default=None,
                    help="v9b Doppler 변화율 잔차 스케일을 κ=0.155에서 이 값으로 바꿔 사용 (1 = 실제 변화율, 민감도 실험용)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    tr, va, ev = (Split(a.data_root, s, a.sc_stride, dev, elev_range=a.elev_range) for s in ("train", "val", a.split))
    if a.doppler_rate_kappa is not None:                                  # Doppler 변화율 스케일 민감도 (v9b 전용)
        for sp in (tr, va, ev):
            rescale_doppler_rate(sp, a.doppler_rate_kappa)
    Tp = a.Tp or tr.Tp_file
    ck = []
    for spec in a.ckpt:
        name, rest = spec.split("=", 1); path, kms = rest.rsplit(":", 1)
        if path.startswith("dl:"):                                          # 딥러닝 baseline (train_baseline_dl.py)
            from models_dl import build_model
            _, mname, path = path.split(":", 2)
            sd = torch.load(path, map_location=dev, weights_only=True)
            G = build_model(mname, Tp).to(dev)
        else:
            sd = torch.load(path, map_location=dev, weights_only=True)
            G = build_model_from_state_dict(sd, Tp, dropout=a.dropout).to(dev)   # 구조는 체크포인트 키로 자동 판별
        G.load_state_dict(sd); G.eval()
        ck.append((name, G, int(round(float(kms) / tr.dt_ms))))
    ks = sorted({int(round(x / tr.dt_ms)) for x in (a.ks_ms or [])} | {c[2] for c in ck})
    Tf = max(max(ks), tr.Tf_file)
    Wtr, Wva, Wev = (Windows(s, Tp, Tf, a.win_stride, a.doppler) for s in (tr, va, ev))
    print(f"eval split={a.split} {ev}  windows={len(Wev)}  ks={ks} (x{tr.dt_ms}ms)  load {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    bl = Baselines(Tp, ks, tr.dt_ms, dev).fit(Wtr, Wva)
    print(f"baselines fitted {time.time()-t0:.1f}s  noise_var={bl.nv:.3e}  LinReg α by k={bl.alpha}  BeamLinReg α by k={bl.alpha_beam}  ManifoldLinReg α by k={bl.alpha_manifold}  ManifoldCombo α by k={bl.alpha_combo}", flush=True)

    methods = list(NAMES) + [c[0] for c in ck]
    nk = len(ks)
    num = {m: [] for m in methods}; den, eid, blk = [], [], []
    probe = {c[0]: [[], []] for c in ck}; shuf = {c[0]: [0.0, 0.0] for c in ck}
    bank = {c[0]: [] for c in ck}           # T2용: 10배치 전 정답 (평가 배치는 에피소드 순서라 다른 에피소드가 되도록)
    t0 = time.time()
    with torch.no_grad():
        for d in Wev.eval_batches(16):
            B, S = d["hist_n"].shape[:2]
            Y = d["fut"][:, :, [k - 1 for k in ks]].to(torch.complex128)                   # (B,S,nk,256)
            den.append((Y.abs() ** 2).sum(dim=(1, 3)).cpu())                                # (B,nk)
            eid.append(d["e"].cpu())
            blk.append(torch.stack([ev.block[d["e"], d["w"] + Tp + k - 1] for k in ks], 1).cpu())
            for m, p in bl.predict(d).items():
                num[m].append(((p - Y).abs() ** 2).sum(dim=(1, 3)).cpu())
            for name, G, kc in ck:
                x, _ = to_model_io(d, kc)
                pc = from_model_out(G(x), B, S).to(torch.complex128)                       # (B,S,256)
                e_all = torch.full((B, nk), float("nan"), dtype=torch.float64)
                if kc in ks:
                    e_all[:, ks.index(kc)] = ((pc - Y[:, :, ks.index(kc)]).abs() ** 2).sum(dim=(1, 2)).cpu()
                num[name].append(e_all)
                if a.leak_check:
                    F = d["fut"].to(torch.complex128)                                       # (B,S,Tf,256)
                    probe[name][0].append(torch.stack([((pc - F[:, :, j]).abs() ** 2).sum(dim=(1, 2)) for j in range(Tf)], 1).cpu())
                    probe[name][1].append((F.abs() ** 2).sum(dim=(1, 3)).cpu())          # (B,Tf) -> pooled 비율로 계산
                    bank[name].append((d["e"].clone(), F[:, :, kc - 1].clone()))
                    if len(bank[name]) > 10:
                        e_old, y_old = bank[name].pop(0)
                        n = min(B, len(e_old)); m = (e_old[:n] != d["e"][:n])            # 다른 에피소드 쌍만
                        if m.any():
                            dif = pc[:n][m] - y_old[:n][m]
                            shuf[name][0] += float((dif.abs() ** 2).sum()); shuf[name][1] += float((y_old[:n][m].abs() ** 2).sum())
    num = {m: torch.cat(v).numpy() for m, v in num.items()}                                 # (Nw, nk)
    den = torch.cat(den).numpy(); eid = torch.cat(eid).numpy(); blk = torch.cat(blk).numpy()
    print(f"evaluated {len(eid)} windows in {time.time()-t0:.1f}s", flush=True)

    el = ev.meta["elevDeg"].astype(float); nlos = ev.nlos
    ep_groups = {"all": np.ones(ev.E, bool), "LOS": ~nlos, "NLOS": nlos, "elev<50": el < 50, "elev>=50": el >= 50}
    db = lambda n, d: 10 * np.log10(np.nansum(n) / np.nansum(d))
    report = {"data_root": a.data_root, "split": a.split, "ks": ks, "dt_ms": tr.dt_ms, "Tp": Tp,
              "linreg_alpha": bl.alpha, "beamlinreg_alpha": bl.alpha_beam, "manifoldlinreg_alpha": bl.alpha_manifold, "manifoldcombo_alpha": bl.alpha_combo, "groups": {}, "paired": {}}
    rng = np.random.default_rng(0)
    for j, k in enumerate(ks):
        cols = [m for m in methods if not np.all(np.isnan(num[m][:, j]))]
        print(f"\n===== horizon k={k} ({k*tr.dt_ms:g} ms)  NMSE [dB] =====")
        print("group              n_win " + "".join(f"{m:>17s}" for m in cols))
        groups = {g: np.isin(eid, np.where(mk)[0]) for g, mk in ep_groups.items()}
        groups["blocked@target"] = blk[:, j] > 3; groups["clear@target"] = blk[:, j] <= 3
        for g, mw in groups.items():
            if mw.sum() == 0:
                continue
            row = {m: db(num[m][mw, j], den[mw, j]) for m in cols}
            report["groups"].setdefault(str(k), {})[g] = {"n_windows": int(mw.sum()), **row}
            print(f"{g:16s} {mw.sum():7d} " + "".join(f"{row[m]:17.2f}" for m in cols))
        # paired bootstrap per RCG model at its own horizon
        E = ev.E
        epnum = {m: np.bincount(eid, num[m][:, j], E) for m in cols}
        epden = np.bincount(eid, den[:, j], E)
        for name, _, kc in ck:
            if kc != k:
                continue
            print(f"  paired (episode bootstrap {a.n_boot}x): Δ = NMSE({name}) - NMSE(base), 음수 = RCG 우위")
            for gname, mk in ep_groups.items():
                idx = np.where(mk & (epden > 0))[0]
                if len(idx) < 3:
                    continue
                sel = idx[rng.integers(0, len(idx), (a.n_boot, len(idx)))]
                for m in NAMES:
                    dobs = db(epnum[name][idx], epden[idx]) - db(epnum[m][idx], epden[idx])
                    dbs = 10 * np.log10(epnum[name][sel].sum(1) / epden[sel].sum(1)) - 10 * np.log10(epnum[m][sel].sum(1) / epden[sel].sum(1))
                    lo, hi = np.percentile(dbs, [2.5, 97.5]); pb = float((dbs >= 0).mean())
                    wins = int((epnum[name][idx] < epnum[m][idx]).sum())
                    report["paired"].setdefault(name, {}).setdefault(gname, {})[m] = {
                        "delta_db": dobs, "ci95": [lo, hi], "p_boot": pb, "rcg_wins": wins, "n_ep": len(idx)}
                    if gname in ("all", "LOS", "NLOS"):
                        sig = "*" if hi < 0 else ("x" if lo > 0 else " ")
                        print(f"    [{gname:4s}] vs {m:17s} Δ={dobs:+6.2f} CI[{lo:+6.2f},{hi:+6.2f}] {sig} p={pb:.4f} wins {wins}/{len(idx)}")
    if a.leak_check:
        for name, _, kc in ck:
            Pn, Pd = torch.cat(probe[name][0]).numpy(), torch.cat(probe[name][1]).numpy()
            curve = 10 * np.log10(Pn.sum(0) / Pd.sum(0))                                      # pooled (다른 지표와 동일)
            report.setdefault("leak_check", {})[name] = {
                "T2_shuffled_target_nmse_db": 10 * np.log10(shuf[name][0] / shuf[name][1]),
                "T3_output_vs_horizon_nmse_db": curve.tolist(), "trained_k": kc}
            print(f"\n[leak check] {name} (trained k={kc}): T2 shuffled-target NMSE = "
                  f"{report['leak_check'][name]['T2_shuffled_target_nmse_db']:.2f} dB (정상: ~0dB 이상)")
            print("  T3 output vs target at k' (최소가 학습 k여야 정상): " +
                  " ".join(f"k'={j+1}:{c:.1f}" for j, c in enumerate(curve) if (j + 1) in sorted({1, kc // 2, kc - 1, kc, min(kc + 1, Tf), Tf})))
    out = a.out or f"eval_{os.path.basename(os.path.normpath(a.data_root))}_{a.split}_{'_'.join(c[0] for c in ck) or 'baselines'}.json"
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
