"""
baselines.py -- 비교 기준 예측기 (전부 GPU 배치, data.Windows 배치를 그대로 입력)
===================================================================================
모든 baseline은 RCG와 같은 정보 조건: 입력 = noisy history(Tp)만, subcarrier마다 독립 처리
(같은 윈도우의 다른 subcarrier 정보 미사용). train/val은 통계·학습용으로만 사용.

  Persistence      : 마지막 history 프레임 (Doppler 잔차 회전 제거 후 = 위상 선형 외삽)
  LMMSE            : train clean 스트림의 시간 자기상관 r(τ)(정상성 가정, 안테나/subcarrier 풀링, horizon k마다 구간 Tp+k) + train 평균 노이즈 분산
  Kalman           : 같은 통계로 만든 스칼라 AR(1) 상태공간 추적
  LinReg_eig       : 학습형 선형(ridge). train 공간 공분산 고유기저 U에서 방향별 시간 필터, 정규화 α는 val로 k별 선택
  Poly2_blind      : subcarrier별 고유빔(history Tp x 256의 top 우특이벡터) + 빔계수 위상 2차/진폭 1차 외삽 (학습 없음)
  BeamLinReg       : 빔 영역 선형 예측. Poly2_blind와 같은 추정 고유빔 + 빔계수에 train으로 학습한 Tp-탭 ridge 필터
                     (α는 val로 k별 선택). 입력 정보는 다른 baseline과 동일 (noisy history만)
  ManifoldLinReg   : BeamLinReg와 같되 빔을 배열 구조(16x16 등간격 평면파)로 추정 (csi_models.manifold_beam:
                     2D FFT 탐색 + 정밀 탐색). = BSC-UNet에서 UNet을 뺀 것 (ablation 겸 비교용)
  ManifoldPoly2    : ManifoldLinReg와 같은 배열 구조 빔 + Poly2_blind와 같은 빔계수 위상 2차/진폭 1차 외삽 (학습 없음)
(필수 비교 5종. ephemeris 보조 참고값(Poly2 oracle)은 v9에서 제외)
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from csi_models import manifold_beam

C128 = torch.complex128
ALPHAS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
NAMES = ["Persistence", "LMMSE", "Kalman", "LinReg_eig", "Poly2_blind", "BeamLinReg", "ManifoldLinReg", "ManifoldPoly2"]


def _autocorr(X):
    """X (..., L) -> Σ_{leading} r(τ) = Σ_t x(t) conj(x(t+τ)), τ=0..L-1."""
    L = X.shape[-1]
    F = torch.fft.fft(X, n=2 * L, dim=-1)
    return torch.fft.ifft(F * F.conj(), dim=-1)[..., :L].reshape(-1, L).to(C128).sum(0).conj()


def _blind_beam(Hs):
    """Hs (N, Tp, 256) noisy history -> 단위 결합 벡터 v (N, 256). 빔계수 s = Hs v, 빔 방향 = conj(v).
    v = Hs^H u / ||.||, u = Gram Hs Hs^H (Tp x Tp)의 top 고유벡터 (= Hs의 top 우특이벡터의 켤레)."""
    _, Uu = torch.linalg.eigh(Hs @ Hs.mH)
    v = torch.einsum("btd,bt->bd", Hs.conj(), Uu[..., -1])
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-30)


def _manifold_v_fn(Hs):
    """Hs (N,Tp,256) -> 결합 벡터 v (N,256): s = Hs v 가 a^H h 가 되도록 v = conj(a), a = 배열 구조 평면파 (단위 노름)."""
    a = manifold_beam(Hs.reshape(Hs.shape[0], Hs.shape[1], 16, 16)).reshape(Hs.shape[0], 256)
    return a.conj().to(Hs.dtype)


def _ridge(A, B, alpha):
    n = A.shape[-1]
    lam = alpha * torch.diagonal(A, dim1=-2, dim2=-1).real.sum(-1) / n
    return torch.linalg.solve(A + lam[..., None, None] * torch.eye(n, dtype=A.dtype, device=A.device), B)


class Baselines:
    def __init__(self, Tp: int, ks: List[int], dt_ms: float, device="cuda"):
        self.Tp, self.ks, self.dt_s, self.dev = Tp, list(ks), dt_ms * 1e-3, device
        self.kmax = max(ks)

    # ------------------------------------------------------------------ fit
    @torch.no_grad()
    def fit(self, train_w, val_w, windows_per_batch: int = 16, seed: int = 1):
        Tp, dev = self.Tp, self.dev
        # 통계 추정 구간은 평가 horizon 집합과 무관하게 고정 (평가 조합에 따라 baseline 값이 바뀌지 않도록):
        #   LMMSE/Kalman 시간 자기상관 = horizon k마다 history+미래 k (길이 Tp+k), LinReg 공간 공분산 = 미래 Tf 전체
        kst = train_w.Tf
        assert kst >= self.kmax, f"Tf={kst} < kmax={self.kmax}"
        r = {k: torch.zeros(Tp + k, dtype=C128, device=dev) for k in self.ks}
        rc = {k: torch.zeros(Tp + k, dtype=torch.float64, device=dev) for k in self.ks}
        Sig = torch.zeros(256, 256, dtype=C128, device=dev)
        nv_num = nv_cnt = 0.0
        Ag = None
        def batches():                                          # 같은 시드로 재생성 (GPU에 쌓아두지 않음)
            g = torch.Generator(device=dev); g.manual_seed(seed)
            return train_w.iterate(windows_per_batch, True, gen=g, perm_gen=torch.Generator().manual_seed(seed),
                                   want_clean_hist=True)
        for d in batches():                                     # pass 1: 시간/공간 통계, 노이즈 분산
            for k in self.ks:
                X = torch.cat([d["hist"], d["fut"][:, :, :k]], 2)                 # clean (B,S,Tp+k,256)
                L = Tp + k
                r[k] += _autocorr(X.transpose(2, 3)); rc[k] += X.shape[0] * X.shape[1] * 256 * (L - torch.arange(L, device=dev)).double()
            F = d["fut"][:, :, :kst].reshape(-1, 256).to(C128)
            Sig += F.T @ F.conj()
            dn = d["hist_n"] - d["hist"]
            nv_num += float((dn.abs() ** 2).sum()); nv_cnt += dn.numel()
        self.r = {k: r[k] / rc[k] for k in self.ks}; self.nv = nv_num / nv_cnt
        _, U = torch.linalg.eigh(Sig); self.U = U.flip(-1)
        # LMMSE / Kalman (스칼라 시간 통계)
        idx = range(Tp)
        I = torch.eye(Tp, dtype=C128, device=dev)
        W, self.K, self.phi = [], {}, {}
        for k in self.ks:                                                                # horizon별 통계
            rk = self.r[k]
            rr = lambda t: rk[t] if t >= 0 else rk[-t].conj()
            R = torch.stack([torch.stack([rr(j - i) for j in idx]) for i in idx])
            W.append(torch.linalg.solve(R + self.nv * I, torch.stack([rr(Tp - 1 + k - i) for i in idx])))
            r0 = rk[0].real.item(); phi = (rk[1].conj() / rk[0].real).item()
            q = r0 * (1 - abs(phi) ** 2); Pv = r0; Ks = []
            for _ in range(Tp):
                K = Pv / (Pv + self.nv); Ks.append(K); Pv = (abs(phi) ** 2) * (1 - K) * Pv + q
            self.K[k], self.phi[k] = Ks, phi
        self.W_lmmse = torch.stack(W)                                                    # (nk, Tp)
        # LinReg_eig: 고유기저 방향별 ridge (train), α는 val로 k별 선택
        Ae = torch.zeros(256, Tp, Tp, dtype=C128, device=dev); Be = torch.zeros(256, Tp, len(self.ks), dtype=C128, device=dev)
        # BeamLinReg: 추정 고유빔 계수 s (Tp) -> 미래 빔계수 (nk), train ridge, α는 val로 k별 선택
        BT = ("svd", "manifold")                                # BeamLinReg / ManifoldLinReg
        Ab = {t: torch.zeros(Tp, Tp, dtype=C128, device=dev) for t in BT}
        Bb = {t: torch.zeros(Tp, len(self.ks), dtype=C128, device=dev) for t in BT}
        Uc = self.U.to(torch.complex64)
        for d in batches():                                     # pass 2 (pass 1과 동일 윈도우·노이즈)
            B_, S = d["hist_n"].shape[:2]
            he = (d["hist_n"].reshape(-1, 256) @ Uc.conj()).reshape(B_ * S, Tp, 256).permute(2, 0, 1)
            ye = (d["fut"][:, :, [k - 1 for k in self.ks]].reshape(-1, 256) @ Uc.conj()).reshape(B_ * S, len(self.ks), 256).permute(2, 0, 1)
            Ae += (he.mH @ he).to(C128); Be += (he.mH @ ye).to(C128)
            for t in BT:
                sb, yb, _, _ = self._beam_coef(d, t)
                Ab[t] += sb.mH @ sb; Bb[t] += sb.mH @ yb
        cands = [_ridge(Ae, Be, a) for a in ALPHAS]                                     # (256, Tp, nk)
        cands_b = {t: [_ridge(Ab[t], Bb[t], a) for a in ALPHAS] for t in BT}             # (Tp, nk)
        vnum = torch.zeros(len(ALPHAS), len(self.ks), dtype=torch.float64, device=dev)
        vnum_b = {t: torch.zeros_like(vnum) for t in BT}
        vden = torch.zeros(len(self.ks), dtype=torch.float64, device=dev)
        for d in val_w.eval_batches(windows_per_batch, seed=777):
            y = d["fut"][:, :, [k - 1 for k in self.ks]].to(C128)                         # (B,S,nk,256)
            vden += (y.abs() ** 2).sum(dim=(0, 1, 3))
            he = d["hist_n"].to(C128) @ self.U.conj()                                   # (B,S,Tp,256)
            for i, W in enumerate(cands):
                p = torch.einsum("bstm,mtk->bskm", he, W) @ self.U.T
                vnum[i] += ((p - y).abs() ** 2).sum(dim=(0, 1, 3))
            for t in BT:
                sb, _, v, Y = self._beam_coef(d, t)
                for i, W in enumerate(cands_b[t]):
                    p = (sb @ W)[..., None] * v.conj()[:, None, :]                      # (N,nk,256)
                    vnum_b[t][i] += ((p - Y).abs() ** 2).sum(dim=(0, 2))
        best = (vnum / vden).argmin(0).cpu().numpy()
        self.W_eig = torch.stack([cands[best[j]][:, :, j] for j in range(len(self.ks))], 2)
        self.alpha = [ALPHAS[b] for b in best]
        self.W_bt, self.alpha_bt = {}, {}
        for t in BT:
            best_b = (vnum_b[t] / vden).argmin(0).cpu().numpy()
            self.W_bt[t] = torch.stack([cands_b[t][best_b[j]][:, j] for j in range(len(self.ks))], 1)   # (Tp, nk)
            self.alpha_bt[t] = [ALPHAS[b] for b in best_b]
        self.W_beam, self.alpha_beam = self.W_bt["svd"], self.alpha_bt["svd"]
        self.alpha_manifold = self.alpha_bt["manifold"]
        return self

    def _beam_coef(self, d, kind="svd"):
        """배치 -> (빔계수 history s (N,Tp), 미래 빔계수 (N,nk), 결합 벡터 v (N,256), 미래 정답 (N,nk,256)), N = B*S.
        kind: 'svd' = history top 특이벡터 (BeamLinReg), 'manifold' = 배열 구조 평면파 (ManifoldLinReg)."""
        Hs = d["hist_n"].to(C128).reshape(-1, self.Tp, 256)
        v = _blind_beam(Hs) if kind == "svd" else self._manifold_v(Hs)
        Y = d["fut"][:, :, [k - 1 for k in self.ks]].to(C128).reshape(Hs.shape[0], len(self.ks), 256)
        return torch.einsum("btd,bd->bt", Hs, v), torch.einsum("bkd,bd->bk", Y, v), v, Y

    @staticmethod
    def _manifold_v(Hs):
        return _manifold_v_fn(Hs)

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, d: Dict) -> Dict[str, torch.Tensor]:
        """d: data.Windows 배치. Returns name -> (B, S, nk, 256) complex128."""
        Tp, ks = self.Tp, self.ks
        h = d["hist_n"].to(C128)                                                         # (B,S,Tp,256)
        B, S = h.shape[:2]
        nk = len(ks)
        out = {"Persistence": h[:, :, -1:].expand(B, S, nk, 256)}
        out["LMMSE"] = torch.einsum("bstd,kt->bskd", h, self.W_lmmse.conj())
        kal = []
        for k in ks:                                                                     # horizon별 AR(1) 통계
            x_prior = torch.zeros_like(h[:, :, 0])
            for t in range(Tp):
                x_post = x_prior + self.K[k][t] * (h[:, :, t] - x_prior); x_prior = self.phi[k] * x_post
            kal.append(self.phi[k] ** k * x_post)
        out["Kalman"] = torch.stack(kal, 2)
        out["LinReg_eig"] = torch.einsum("bstm,mtk->bskm", h @ self.U.conj(), self.W_eig) @ self.U.T
        # Poly2: subcarrier별 고유빔 (Tp x 256 -> Gram Tp x Tp로 top 특이벡터)
        Hs = h.reshape(B * S, Tp, 256)
        v = _blind_beam(Hs)                                                              # (BS, 256)
        s = torch.einsum("btd,bd->bt", Hs, v)                                             # 빔 계수 (BS, Tp)
        t = (torch.arange(Tp, device=h.device) - (Tp - 1)).double()
        tk = torch.tensor(ks, device=h.device, dtype=torch.float64)
        ang = torch.angle(s)
        dd = ang[:, 1:] - ang[:, :-1]
        ddm = torch.remainder(dd + math.pi, 2 * math.pi) - math.pi
        ph = torch.cat([ang[:, :1], ang[:, :1] + torch.cumsum(ddm, -1)], -1)            # unwrap
        V = lambda x, deg: torch.stack([x ** i for i in range(deg, -1, -1)], -1)
        P1, P2 = torch.linalg.pinv(V(t, 1)), torch.linalg.pinv(V(t, 2))
        amp = torch.clamp((s.abs() @ P1.T) @ V(tk, 1).T, min=0)                          # (BS, nk)
        sk = amp * torch.exp(1j * ((ph @ P2.T) @ V(tk, 2).T))                            # (BS, nk)
        out["Poly2_blind"] = (sk[..., None] * v.conj()[:, None, :]).reshape(B, S, nk, 256)
        sh = torch.einsum("btd,bd->bt", Hs, v) @ self.W_beam                              # (BS, nk) 같은 빔
        out["BeamLinReg"] = (sh[..., None] * v.conj()[:, None, :]).reshape(B, S, nk, 256)
        vm = _manifold_v_fn(Hs)
        sm = torch.einsum("btd,bd->bt", Hs, vm) @ self.W_bt["manifold"]
        out["ManifoldLinReg"] = (sm[..., None] * vm.conj()[:, None, :]).reshape(B, S, nk, 256)
        s_m = torch.einsum("btd,bd->bt", Hs, vm)                                          # 매니폴드 빔계수 history
        ang_m = torch.angle(s_m)
        ddm_m = torch.remainder(ang_m[:, 1:] - ang_m[:, :-1] + math.pi, 2 * math.pi) - math.pi
        ph_m = torch.cat([ang_m[:, :1], ang_m[:, :1] + torch.cumsum(ddm_m, -1)], -1)
        amp_m = torch.clamp((s_m.abs() @ P1.T) @ V(tk, 1).T, min=0)
        sk_m = amp_m * torch.exp(1j * ((ph_m @ P2.T) @ V(tk, 2).T))
        out["ManifoldPoly2"] = (sk_m[..., None] * vm.conj()[:, None, :]).reshape(B, S, nk, 256)
        return out
