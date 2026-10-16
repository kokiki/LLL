"""
diagnose.py -- 제안 기법 이득의 출처 진단 (평가 배치는 evaluate.py와 동일: 고정 시드 12345, 16 윈도우/배치)
===============================================================================
(1) 빔 영역 선형 예측기 분해 (빔 방향 x 시간 필터). 예측 = v · ŝ_{t+k},  ŝ = Tp-탭 선형 필터(v^H h_noisy history)
      빔 v   : Ora   = (에피소드, subcarrier)별 clean 스트림 공분산의 주 고유벡터 (오라클)
               Blind = 윈도우별 noisy history (Tp x 256)의 top 우특이벡터 (추정, 구현 가능)
      필터   : Ep    = 같은 test 에피소드의 평가 윈도우 전체로 in-sample 최소제곱 (오라클 통계, 낙관적 상한)
               EpCV  = Ep와 같지만 에피소드 앞/뒤 절반 윈도우로 교차 적합 (한쪽 절반으로 적합, 다른 절반에서 평가 -> in-sample 낙관 제거)
               Tr    = train 전체로 적합한 공통 필터 (구현 가능)
    Ora+Ep = 선형 예측의 낙관적 상한,  Blind+Tr = 오라클 없는 실용 빔 영역 선형 예측기 (추가 baseline 후보)
(2) SNR 구간별 성능: SNR을 고정해 같은 윈도우를 평가 -> SNR별 제안 기법 vs LinReg/Poly2 이득

Usage:
  python diagnose.py --data_root <v9> --split test --ckpt Prop_k4=runs/v9_Tp20_k4ms/best_G.pt:4 [--ckpt ...] --out runs/diag_v9_test.json
"""
import argparse
import json

import numpy as np
import torch

from baselines import Baselines
from data import Split, Windows, from_model_out, to_model_io
from csi_models import build_model_from_state_dict

C128 = torch.complex128
BEAMS = ["Ora+Ep", "Ora+EpCV", "Ora+Tr", "Blind+Ep", "Blind+EpCV", "Blind+Tr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["test", "guard", "val"])
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH:K_MS")
    ap.add_argument("--Tp", type=int, default=20)
    ap.add_argument("--snrs", type=float, nargs="*", default=[5, 10, 15, 20])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    tr, va, ev = (Split(a.data_root, s, device=dev) for s in ("train", "val", a.split))
    Tp = a.Tp
    ck = []
    for spec in a.ckpt:
        name, rest = spec.split("=", 1); path, kms = rest.rsplit(":", 1)
        sd = torch.load(path, map_location=dev, weights_only=True)
        G = build_model_from_state_dict(sd, Tp).to(dev)
        G.load_state_dict(sd); G.eval()
        ck.append((name, G, int(round(float(kms) / tr.dt_ms))))
    ks = sorted({c[2] for c in ck})
    nk = len(ks)
    Tf = max(max(ks), tr.Tf_file)
    Wtr, Wva, Wev = (Windows(s, Tp, Tf) for s in (tr, va, ev))
    bl = Baselines(Tp, ks, tr.dt_ms, dev).fit(Wtr, Wva)
    E, S = ev.E, ev.n_sc
    groups = {"all": np.ones(E, bool), "LOS": ~ev.nlos, "NLOS": ev.nlos}
    db = lambda n, d: float(10 * np.log10(n.sum() / d.sum()))
    eyeT = torch.eye(Tp, dtype=C128, device=dev)
    report = {"split": a.split, "Tp": Tp, "ks": ks}

    def oracle_beams(sp):
        V = torch.empty(sp.E, sp.n_sc, 256, dtype=C128, device=dev)
        for e in range(sp.E):
            Xe = sp.X[e].to(C128)                                               # (S, T, 256)
            _, U = torch.linalg.eigh(Xe.mT @ Xe.conj())                          # Σ_t h h^H (sc별)
            V[e] = U[..., -1]
        return V

    def blind_beam(h):                                                          # (B,S,Tp,256) noisy -> (B,S,256)
        B_, S_ = h.shape[:2]
        Hs = h.reshape(B_ * S_, Tp, 256)
        _, Uu = torch.linalg.eigh(Hs @ Hs.mH)
        v = torch.einsum("btd,bt->bd", Hs.conj(), Uu[..., -1]).conj()           # top 우특이벡터 (위상 임의, v v^H 불변)
        v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-30)
        return v.reshape(B_, S_, 256)

    def coef(d, v):
        s_h = torch.einsum("bstd,bsd->bst", d["hist_n"].to(C128), v.conj())     # (B,S,Tp)
        Y = d["fut"][:, :, [k - 1 for k in ks]].to(C128)                         # (B,S,nk,256)
        return s_h, torch.einsum("bskd,bsd->bsk", Y, v.conj()), Y

    def solve(A, b):
        lam = 1e-6 * torch.diagonal(A, dim1=-2, dim2=-1).real.mean(-1)[..., None, None]
        return torch.linalg.solve(A + lam * eyeT, b[..., None])[..., 0]

    with torch.no_grad():
        # train 공통 필터 (Tr)
        Vtr = oracle_beams(tr)
        At = {b: torch.zeros(nk, Tp, Tp, dtype=C128, device=dev) for b in ("Ora", "Blind")}
        bt = {b: torch.zeros(nk, Tp, dtype=C128, device=dev) for b in ("Ora", "Blind")}
        for d in Wtr.eval_batches(16, seed=1):
            for b, v in (("Ora", Vtr[d["e"]]), ("Blind", blind_beam(d["hist_n"].to(C128)))):
                s_h, s_y, _ = coef(d, v)
                x = s_h.reshape(-1, Tp)
                At[b] += (x.conj().T @ x)[None]
                bt[b] += (x.conj().T @ s_y.reshape(-1, nk)).T
        Wtrf = {b: solve(At[b], bt[b]) for b in At}                              # (nk, Tp)
        del Vtr
        # test 에피소드별 in-sample 필터 (Ep)
        V = oracle_beams(ev)
        Wep, Wcv = {}, {}
        half_w = (ev.T - Tp - Tf + 1) // 2
        for b in ("Ora", "Blind"):
            A = torch.zeros(2 * E, S, nk, Tp, Tp, dtype=C128, device=dev)       # 인덱스 = 2e + (윈도우가 뒤쪽 절반?)
            bb = torch.zeros(2 * E, S, nk, Tp, dtype=C128, device=dev)
            for d in Wev.eval_batches(16):
                e = d["e"]; ih = 2 * e + (d["w"] >= half_w).long()
                s_h, s_y, _ = coef(d, V[e] if b == "Ora" else blind_beam(d["hist_n"].to(C128)))
                outer = s_h.conj()[..., :, None] * s_h[..., None, :]
                for j in range(nk):
                    A[:, :, j].index_add_(0, ih, outer)
                    bb[:, :, j].index_add_(0, ih, s_h.conj() * s_y[:, :, j, None])
            Wep[b] = solve(A[0::2] + A[1::2], bb[0::2] + bb[1::2])               # (E,S,nk,Tp) 전체 in-sample
            Wcv[b] = solve(A, bb)                                                # (2E,...) 절반별 -> 반대쪽 절반에 적용
            del A, bb

        def run(snr=None, with_beams=True):
            names = ["LinReg_eig", "Poly2_blind"] + (BEAMS if with_beams else []) + [c[0] for c in ck]
            num = {m: torch.zeros(E, nk, dtype=torch.float64) for m in names}
            den = torch.zeros(E, nk, dtype=torch.float64)
            for d in Wev.eval_batches(16, snr_db=snr):
                e = d["e"]; ec = e.cpu(); B = len(e)
                Y = d["fut"][:, :, [k - 1 for k in ks]].to(C128)
                den.index_add_(0, ec, (Y.abs() ** 2).sum(dim=(1, 3)).cpu())
                bp = bl.predict(d)
                for m in ("LinReg_eig", "Poly2_blind"):
                    num[m].index_add_(0, ec, ((bp[m] - Y).abs() ** 2).sum(dim=(1, 3)).cpu())
                if with_beams:
                    for b, v in (("Ora", V[e]), ("Blind", blind_beam(d["hist_n"].to(C128)))):
                        s_h, _, _ = coef(d, v)
                        other = 2 * e + 1 - (d["w"] >= half_w).long()               # 반대쪽 절반으로 적합한 필터
                        for f, sh in (("Ep", torch.einsum("bst,bskt->bsk", s_h, Wep[b][e])),
                                      ("EpCV", torch.einsum("bst,bskt->bsk", s_h, Wcv[b][other])),
                                      ("Tr", torch.einsum("bst,kt->bsk", s_h, Wtrf[b]))):
                            p = sh[..., None] * v[:, :, None, :]
                            num[f"{b}+{f}"].index_add_(0, ec, ((p - Y).abs() ** 2).sum(dim=(1, 3)).cpu())
                for name, G, kc in ck:
                    x, _ = to_model_io(d, kc)
                    pc = from_model_out(G(x), B, S).to(C128)
                    err = torch.zeros(B, nk, dtype=torch.float64)
                    err[:, ks.index(kc)] = ((pc - Y[:, :, ks.index(kc)]).abs() ** 2).sum(dim=(1, 2)).cpu()
                    num[name].index_add_(0, ec, err)
            return names, num, den

        names, num, den = run()
        print("\n===== (1) 빔 영역 선형 예측기 분해 NMSE [dB]  (Ora=오라클 빔 / Blind=추정 빔,  Ep=에피소드별 in-sample 필터 / Tr=train 공통 필터) =====")
        cols = ["LinReg_eig", "Poly2_blind"] + BEAMS
        short = {"LinReg_eig": "LinReg", "Poly2_blind": "Poly2"}
        report["beam"] = {}
        for name, _, kc in ck:
            j = ks.index(kc)
            print(f"\n  k={kc}")
            print("  group  " + "".join(f"{short.get(m, m):>10s}" for m in cols) + f"{'Prop':>10s}")
            for g, mk in groups.items():
                r = {m: db(num[m][mk, j].numpy(), den[mk, j].numpy()) for m in cols + [name]}
                report["beam"].setdefault(str(kc), {})[g] = r
                print(f"  {g:5s} " + "".join(f"{r[m]:10.2f}" for m in cols + [name]))

        if a.snrs:
            print("\n===== (2) SNR 고정 평가: 제안 기법 NMSE (LinReg 대비 Δ, Poly2 대비 Δ) [dB] =====")
            report["snr"] = {}
            for snr in a.snrs:
                _, n2, d2 = run(snr, with_beams=False)
                for name, _, kc in ck:
                    j = ks.index(kc)
                    for g, mk in groups.items():
                        r = {m: db(n2[m][mk, j].numpy(), d2[mk, j].numpy()) for m in ("LinReg_eig", "Poly2_blind", name)}
                        report["snr"].setdefault(str(kc), {}).setdefault(g, {})[f"{snr:g}"] = r
            for name, _, kc in ck:
                print(f"\n  k={kc}")
                print("  group " + "".join(f"        SNR={s:<4g}        " for s in a.snrs))
                for g in groups:
                    cells = []
                    for s in a.snrs:
                        r = report["snr"][str(kc)][g][f"{s:g}"]
                        cells.append(f"{r[name]:7.2f} ({r[name]-r['LinReg_eig']:+5.2f},{r[name]-r['Poly2_blind']:+5.2f})")
                    print(f"  {g:5s} " + "  ".join(cells))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)
        print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
