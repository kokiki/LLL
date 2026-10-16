"""
robust_calib.py -- 배열 보정 오차의 온라인(과거 관측 기반) 추정·보상 시험
===============================================================================
관측 채널 = c ⊙ h (c: 소자별 고정 복소 이득 오차, 배열 1개 -> 모든 에피소드·시간 공통). compare_models.py의 (2)와 같은 오차 모델.
  ĉ 추정 (정답·clean 채널 미사용): test 이전 운용 데이터(= val 에피소드에 같은 c를 적용한 noisy history)의 윈도우마다
    b = history top 특이벡터 방향(∝ c⊙a),  a = 배열 구조 평면파 (models_beam.manifold_beam),  q = b ⊘ a
    윈도우별 공통 위상·크기를 맞춘 뒤 평균 -> ĉ  (전역 복소 상수는 무의미하므로 평균 |ĉ| = 1, 위상 기준 임의)
  보상: 입력 history와 정답을 ĉ로 나눈 영역에서 예측 -> 관측 영역 오차 = Σ_i |ĉ_i|² |p_i - y_i|² 로 평가
    (근사: 노이즈는 보상 후 영역에서 주입 -> 소자별 노이즈 착색 무시, 이득 오차 ≤2dB에서 영향 작음)
비교: 보상 없음 / 추정 ĉ 보상 (val 윈도우 수 n_cal별) / 참값 c 보상 (오라클 참고)
"""
import argparse
import json

import numpy as np
import torch

from baselines import Baselines, _blind_beam
from compare_models import ROBUST_LEVELS, load
from data import Split, Windows, from_model_out, to_model_io
from models_beam import manifold_beam

C128 = torch.complex128
BASES = ["LinReg_eig", "Poly2_blind", "ManifoldLinReg"]


@torch.no_grad()
def estimate_c(Wva, n_windows):
    """val noisy history로 ĉ (256,) 추정. n_windows: 사용할 윈도우 수 (각 윈도우 x subcarrier가 표본)."""
    acc = torch.zeros(256, dtype=C128, device="cuda"); used = 0
    for d in Wva.eval_batches(16, seed=4242):
        Hs = d["hist_n"].to(C128).reshape(-1, Wva.Tp, 256)                    # (N, Tp, 256)
        b = _blind_beam(Hs).conj()                                            # 빔 방향 ∝ c⊙a
        a = manifold_beam(Hs.reshape(-1, Wva.Tp, 16, 16)).reshape(-1, 256).to(C128)
        q = b / a                                                             # ∝ c (윈도우별 복소 상수 배)
        ref = q.sum(-1, keepdim=True)
        q = q * ref.conj() / ref.abs().clamp_min(1e-30)                       # 윈도우별 공통 위상 제거
        q = q / q.abs().mean(-1, keepdim=True).clamp_min(1e-30)               # 윈도우별 크기 정규화
        if used == 0:
            acc += q.sum(0)
        else:                                                                 # 누적 추정치 위상에 맞춰 정렬 후 더함
            ph = (q * acc.conj()).sum(-1, keepdim=True)
            acc += (q * ph.conj() / ph.abs().clamp_min(1e-30)).sum(0)
        used += len(d["e"])
        if used >= n_windows:
            break
    c = acc / acc.abs().mean()
    return c, used


@torch.no_grad()
def eval_comp(Wev, ev, ks, models, bl, chat):
    """보상 영역(ev.X = X0*c/ĉ로 이미 설정)에서 예측, 관측 영역 가중(|ĉ|²)으로 pooled 오차 누적 -> {m: (E,nk)}, den."""
    E, nk = ev.E, len(ks)
    w = (chat.abs() ** 2).to(torch.float64)                                   # (256,)
    names = list(models) + BASES
    num = {m: np.zeros((E, nk)) for m in names}; den = np.zeros((E, nk))
    for d in Wev.eval_batches(16):
        e = d["e"].cpu().numpy(); B, S = d["hist_n"].shape[:2]
        Y = d["fut"][:, :, [k - 1 for k in ks]].to(C128)
        for j in range(nk):
            np.add.at(den[:, j], e, ((Y[:, :, j].abs() ** 2) * w).sum(dim=(1, 2)).cpu().numpy())
        for name, Gk in models.items():
            for j, k in enumerate(ks):
                x, _ = to_model_io(d, k)
                p = from_model_out(Gk[k](x), B, S).to(C128)
                np.add.at(num[name][:, j], e, (((p - Y[:, :, j]).abs() ** 2) * w).sum(dim=(1, 2)).cpu().numpy())
        bp = bl.predict(d)
        for m in BASES:
            for j in range(nk):
                np.add.at(num[m][:, j], e, (((bp[m][:, :, j] - Y[:, :, j]).abs() ** 2) * w).sum(dim=(1, 2)).cpu().numpy())
    return num, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pair", action="append", required=True, help="NAME=PATH_TEMPLATE ({K})")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--Tp", type=int, default=20)
    ap.add_argument("--n_cal", type=int, nargs="+", default=[160, 1600, 6900], help="ĉ 추정에 쓰는 val 윈도우 수")
    ap.add_argument("--n_draw", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    tr, va, ev = (Split(a.data_root, s, device=dev) for s in ("train", "val", a.split))
    Tp, ks = a.Tp, a.ks
    Tf = max(max(ks), tr.Tf_file)
    models = {p.split("=", 1)[0]: {k: load(p.split("=", 1)[1].replace("{K}", str(k)), Tp, dev) for k in ks} for p in a.pair}
    Wtr, Wva, Wev = Windows(tr, Tp, Tf), Windows(va, Tp, Tf), Windows(ev, Tp, Tf)
    bl = Baselines(Tp, ks, tr.dt_ms, dev).fit(Wtr, Wva)                       # 이상적 배열 train으로 적합
    X0, V0 = ev.X.clone(), va.X.clone()
    groups = {"all": np.ones(ev.E, bool), "LOS": ~ev.nlos, "NLOS": ev.nlos}
    db = lambda n, d: float(10 * np.log10(n.sum() / d.sum()))
    names = list(models) + BASES
    report = {}
    for ph_deg, g_db in ROBUST_LEVELS[1:]:
        lvl = f"phase{ph_deg:g}deg_gain{g_db:g}dB"
        modes = ["none"] + [f"est{n}" for n in a.n_cal] + ["oracle"]
        acc = {mo: None for mo in modes}; cerr = {mo: [] for mo in modes}
        for r in range(a.n_draw):
            gen = torch.Generator(device="cpu").manual_seed(1000 + r)              # compare_models.py와 같은 오차 실현
            phi = torch.randn(256, generator=gen) * np.deg2rad(ph_deg)
            amp = 10 ** (torch.randn(256, generator=gen) * g_db / 20)
            c = (amp * torch.exp(1j * phi)).to(torch.complex64).to(dev)
            va.X = V0 * c                                                          # 과거 운용 데이터도 같은 배열
            for mo in modes:
                if mo == "none":
                    chat = torch.ones(256, dtype=torch.complex64, device=dev)
                elif mo == "oracle":
                    chat = c
                else:
                    ch, _ = estimate_c(Wva, int(mo[3:]))
                    ph = (ch * c.conj().to(C128)).sum(); ch = ch * ph.conj() / ph.abs()   # 보고용 정렬 (전역 상수는 무의미)
                    scale = (c.to(C128).abs().mean() / ch.abs().mean()); ch = ch * scale
                    cerr[mo].append(float(10 * np.log10(((ch - c.to(C128)).abs() ** 2).sum().item() / (c.abs() ** 2).sum().item())))
                    chat = ch.to(torch.complex64)
                ev.X = X0 * (c / chat)                                             # 보상 영역의 관측
                n_, d_ = eval_comp(Wev, ev, ks, models, bl, chat)
                acc[mo] = (n_, d_) if acc[mo] is None else ({m: acc[mo][0][m] + n_[m] for m in names}, acc[mo][1] + d_)
        ev.X, va.X = X0, V0
        for mo in modes:
            for j, k in enumerate(ks):
                for g, mk in groups.items():
                    report.setdefault(lvl, {}).setdefault(mo, {}).setdefault(str(k), {})[g] = {m: db(acc[mo][0][m][mk, j], acc[mo][1][mk, j]) for m in names}
            if cerr[mo]:
                report[lvl][mo]["c_est_error_db"] = float(np.mean(cerr[mo]))
        print(f"\n===== 배열 오차 위상 {ph_deg:g}° / 이득 {g_db:g}dB rms: 전체 NMSE [dB] =====")
        print("  보상          " + "".join(f"  k={k}:" + "".join(f"{m[:8]:>9s}" for m in names) for k in ks))
        for mo in modes:
            ce = f" (c_hat 오차 {report[lvl][mo]['c_est_error_db']:.1f}dB)" if mo.startswith("est") else ""
            row = "".join("      " + "".join(f"{report[lvl][mo][str(k)]['all'][m]:9.2f}" for m in names) for k in ks)
            print(f"  {mo:12s}{row}{ce}", flush=True)
        if a.out:                                                                  # 단계마다 저장
            json.dump(report, open(a.out, "w"), indent=2, default=float)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2, default=float)
        print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
