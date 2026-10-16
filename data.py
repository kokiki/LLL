"""
data.py -- 에피소드 로딩(.mat -> GPU) + GPU 배치 윈도우 생성
===============================================================
MATLAB 생성기(generate_ntn_csi_v9_fdd_dl.m)가 저장한 에피소드(.mat, -v7.3)를 읽어 **처음부터 GPU에 상주**시키고,
윈도우 생성(causal 정규화, SNR 노이즈, Doppler 잔차 회전 제거)을 GPU 배치로 수행한다. 학습/검증/평가/baseline이
모두 이 모듈 하나를 쓰므로 정의가 한 곳에서만 관리된다.

윈도우 정의 (에피소드 e, 시작 w):
  history = 관측 w .. w+Tp-1 (noisy),  미래 = w+Tp .. w+Tp+Tf-1 (clean),  horizon k = 미래 k번째 관측
  1) causal 정규화: history 마지막 관측까지의 running max(causalMaxPerSlot)로 나눔
  2) 노이즈: SNR ~ U[5,20] dB, 노이즈 전력 = **history 신호전력** x 10^(-SNR/10)  (완전 인과적)
  3) Doppler 잔차 회전 제거(doppler='est'): noisy history의 lag-1 공통 회전율 ω를 추정해
     history 오프셋 -(Tp-1)..0, 미래 오프셋 1..Tf에 e^{-jωo} (입력/정답 동일 -> 예측 관계 보존, 인과적)
평가 배치는 고정 시드 + 고정 순서로 생성 -> 같은 스크립트 안에서 RCG와 모든 baseline이 **동일 윈도우·동일 노이즈**.

저장 형식 메모: h5py는 MATLAB 열우선 배열을 축이 뒤집힌 채로 읽으므로 전치해 MATLAB 논리 순서
[Nsc, T, rows, cols]로 맞추고, 안테나는 (rows, cols) C순서로 펼쳐 256차원. dsReal/dsImag는 int16 + dsScale 역양자화.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch

C64 = torch.complex64
META_SCALARS = ("elevDeg", "fDdot_sat", "fD_sat", "arrayBearingDeg", "KdB", "delaySpread", "shadowFadeDB",
                "snapshotIntervalMs", "Tp", "Tf", "satAlt", "satVel")


def _scalar(ds):
    a = np.asarray(ds[()])
    if a.dtype.kind in ("u", "i") and a.size > 1 and a.ndim <= 2 and np.all((a >= 9) & (a <= 126)):
        return "".join(chr(int(c)) for c in a.ravel())          # MATLAB char -> str
    return a.reshape(-1)[0].item() if a.size == 1 else a.ravel()


def read_episode(path, sc_idx: Sequence[int]) -> Dict:
    """에피소드 .mat 1개 -> dict(X: complex64 (n_sc, T, 256), cm: (T,), meta...)."""
    with h5py.File(path, "r") as f:
        rd = lambda k: np.asarray(f[k][()]).transpose(tuple(range(f[k].ndim))[::-1])
        re_, im_ = rd("dsReal")[sc_idx], rd("dsImag")[sc_idx]            # (n_sc, T, rows, cols)
        scale = float(_scalar(f["dsScale"])) if "dsScale" in f else None
        out = {k: _scalar(f[k]) for k in META_SCALARS if k in f}
        out["profile"] = _scalar(f["profile"])
        out["cm"] = np.asarray(f["causalMaxPerSlot"][()], dtype=np.float64).ravel()
        out["blockAttenDB"] = (np.asarray(f["blockAttenDB"][()], dtype=np.float32).ravel()
                               if "blockAttenDB" in f else np.zeros(re_.shape[1], np.float32))
    re_, im_ = re_.astype(np.float32), im_.astype(np.float32)
    if scale is not None:
        re_, im_ = re_ / 32767.0 * scale, im_ / 32767.0 * scale
    n_sc, T = re_.shape[:2]
    out["X"] = (re_ + 1j * im_).astype(np.complex64).reshape(n_sc, T, -1)
    out["name"] = Path(path).name
    return out


N_FFT, SC_CENTER = 512, 150      # 300 subcarrier(25 RB) 반송파의 FFT 크기와 중심 subcarrier 인덱스


def fix_timing_jumps(X: torch.Tensor, sc_idx, max_shift: int = 10, margin: float = 1.2, min_coh: float = 0.75):
    """v9 생성기의 관측별 타이밍 재추정(nrPerfectTimingEstimate)으로 생긴 정수 샘플 기준 변화를 되돌린다.
    기준이 Δ샘플 바뀌면 그 이후 관측 전체에 subcarrier 선형 위상 기울기 e^{-j2π(k-150)Δ/512}가 곱해진다.
    스텝 t-1 -> t의 subcarrier별 lag-1 상관 r_k에 대해 Δ ∈ [-max_shift, max_shift]의 기울기를 맞춰 보고
    (score(Δ) = |Σ_k r_k e^{+j2π(k-150)Δ/512}|), 최적 Δ≠0이 (i) Δ=0보다 margin배 이상 크고 (ii) 정합도
    score/Σ|r_k| ≥ min_coh (기울기 제거 후 subcarrier 위상이 한 값으로 모임)일 때 오프셋 변화로 판정.
    (점프 스텝: 정합도 ≈ 1, 최적/Δ0 ≈ 1.46 / 정상 스텝: 이웃 Δ/Δ0 ≈ 0.69) 누적 Δ로 이후 관측을 보정.
    X: (E, n_sc, T, 256) complex. Returns (보정된 X, 스텝별 Δ (E, T) int)."""
    k = torch.as_tensor(np.asarray(sc_idx), device=X.device, dtype=torch.float64) - SC_CENTER
    r = (X[:, :, 1:] * X[:, :, :-1].conj()).sum(-1).to(torch.complex128)              # (E, S, T-1)
    shifts = torch.arange(-max_shift, max_shift + 1, device=X.device, dtype=torch.float64)
    ramp = torch.exp(1j * 2 * np.pi * k[None, :] * shifts[:, None] / N_FFT)            # (nΔ, S): 기울기 제거용
    score = torch.einsum("ast,ds->adt", r, ramp).abs()                                   # (E, nΔ, T-1)
    best = score.argmax(1)
    s0 = score[:, max_shift]                                                             # Δ=0
    sb = score.gather(1, best[:, None]).squeeze(1)
    coh = sb / r.abs().sum(1).clamp_min(1e-30)
    dlt = shifts[best] * ((sb > margin * s0) & (coh >= min_coh))                         # (E, T-1)
    D = torch.cat([torch.zeros_like(dlt[:, :1]), torch.cumsum(dlt, 1)], 1)               # (E, T) 누적 오프셋
    corr = torch.exp(1j * 2 * np.pi * k[None, :, None] * D[:, None, :] / N_FFT)          # (E, S, T)
    return (X * corr[..., None].to(X.dtype)), torch.cat([torch.zeros_like(dlt[:, :1]), dlt], 1).long()


class Split:
    """split 폴더 하나의 모든 에피소드 (GPU 상주).
    fix_timing=True (기본): v9 생성기의 타이밍 기준 점프를 로드 시 보정 (fix_timing_jumps). False = 원본 그대로."""

    def __init__(self, root, split: str, sc_stride: int = 25, device: str = "cuda", cache: bool = True,
                 elev_range=None, fix_timing: bool = True):
        d = Path(root) / split
        paths = sorted(d.glob("episode_*.mat"))
        if not paths:
            raise FileNotFoundError(f"{d}: episode_*.mat 없음")
        sc_idx = np.arange(0, 300, sc_stride)
        key = hashlib.sha256(repr([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]).encode()
                             + sc_idx.tobytes()).hexdigest()[:12]
        cpath = d / f"_ntncache_sc{len(sc_idx)}_{key}.pt"
        if cache and cpath.exists():
            blob = torch.load(cpath, weights_only=False)
        else:
            eps = [read_episode(p, sc_idx) for p in paths]
            T = min(e["X"].shape[1] for e in eps)
            blob = {"X": torch.from_numpy(np.stack([e["X"][:, :T] for e in eps])),
                    "cm": torch.from_numpy(np.stack([e["cm"][:T] for e in eps])),
                    "block": torch.from_numpy(np.stack([e["blockAttenDB"][:T] for e in eps])),
                    "meta": {k: np.array([e.get(k, np.nan) for e in eps]) for k in META_SCALARS},
                    "profile": np.array([e["profile"] for e in eps]), "names": [e["name"] for e in eps]}
            if cache:
                torch.save(blob, cpath)
        if elev_range is not None:                              # 고각 범위로 에피소드 부분집합 (비교 실험용)
            keep = np.where((blob["meta"]["elevDeg"] >= elev_range[0]) & (blob["meta"]["elevDeg"] < elev_range[1]))[0]
            blob = {"X": blob["X"][keep], "cm": blob["cm"][keep], "block": blob["block"][keep],
                    "meta": {k: v[keep] for k, v in blob["meta"].items()}, "profile": blob["profile"][keep],
                    "names": [blob["names"][i] for i in keep]}
        self.device = torch.device(device)
        self.X = blob["X"].to(self.device)                      # (E, n_sc, T, 256) complex64
        self.sc_idx = sc_idx
        self.timing_shifts = None
        if fix_timing:
            self.X, self.timing_shifts = fix_timing_jumps(self.X, sc_idx)
        self.cm = blob["cm"].to(self.device).double()           # (E, T)
        self.block = blob["block"].to(self.device)              # (E, T) 주 경로 블로키지 감쇠 [dB] (분석용 메타)
        self.meta = blob["meta"]
        self.nlos = blob["profile"] == "NTN-CDL-A"
        self.names = blob["names"]
        self.E, self.n_sc, self.T = self.X.shape[:3]
        self.dt_ms = float(self.meta["snapshotIntervalMs"][0])
        self.Tp_file, self.Tf_file = int(self.meta["Tp"][0]), int(self.meta["Tf"][0])

    def __repr__(self):
        return (f"Split(E={self.E}, n_sc={self.n_sc}, T={self.T}, dt={self.dt_ms}ms, NLOS={int(self.nlos.sum())}, "
                f"Tp_file={self.Tp_file}, Tf_file={self.Tf_file})")


class Windows:
    """한 Split에서 (에피소드, 시작) 윈도우를 GPU 배치로 생성."""

    def __init__(self, sp: Split, Tp: int, Tf: int, stride: int = 1, doppler: str = "est"):
        # est_parity: v7 이전 데이터(SCS60kHz, 0.25ms 관측) 전용 -- 생성기의 고정 슬롯길이 Doppler 보상으로 생긴
        # 짝/홀 슬롯 위상 오프셋까지 추정·제거. v9 데이터(1ms 관측, sampleTimes 보상)는 오프셋이 없으므로 'est'.
        assert doppler in ("est", "est_parity", "none")
        self.sp, self.Tp, self.Tf, self.doppler = sp, Tp, Tf, doppler
        nw = sp.T - Tp - Tf + 1
        assert nw > 0, f"T={sp.T} < Tp+Tf={Tp + Tf}"
        e, w = np.meshgrid(np.arange(sp.E), np.arange(0, nw, stride), indexing="ij")
        self.index = torch.from_numpy(np.stack([e.ravel(), w.ravel()], 1)).to(sp.device)
        dev = sp.device
        self.ar = torch.arange(Tp + Tf, device=dev)
        self.o_h = (torch.arange(Tp, device=dev) - (Tp - 1)).double()
        self.o_f = torch.arange(1, Tf + 1, device=dev).double()

    def __len__(self):
        return len(self.index)

    def batch(self, rows: torch.Tensor, gen: Optional[torch.Generator] = None, snr_db: Optional[float] = None,
              want_clean_hist: bool = False, elem_resp_fn=None) -> Dict:
        """rows: self.index 행 번호 (B,). Returns dict (전부 GPU):
           hist_n (B,n_sc,Tp,256) c64 noisy, fut (B,n_sc,Tf,256) c64 clean, [hist (clean)], e, w, omega.
           elem_resp_fn(B, device, gen) -> (B, 256) 복소: 윈도우별 위성 배열 소자 응답(보정 오차·고장)을 노이즈 주입 전
           채널(history·정답 모두)에 곱함 (학습 증강용, 기본 None = 이상적 배열)."""
        sp, Tp, Tf = self.sp, self.Tp, self.Tf
        e, w = self.index[rows, 0], self.index[rows, 1]
        B = len(e)
        x = sp.X[e[:, None], :, w[:, None] + self.ar].permute(0, 2, 1, 3)        # (B, n_sc, Tp+Tf, 256)
        nf = sp.cm[e, w + Tp - 1]
        nf = torch.where(nf > 0, nf, torch.ones_like(nf))
        x = x / nf.float()[:, None, None, None]
        if elem_resp_fn is not None:
            x = x * elem_resp_fn(B, sp.device, gen)[:, None, None, :]
        hist, fut = x[:, :, :Tp], x[:, :, Tp:]
        sig = (hist.real.double() ** 2 + hist.imag.double() ** 2).mean(dim=(1, 2, 3))   # history 전력 (인과적)
        snr = 5.0 + 15.0 * torch.rand(B, device=sp.device, dtype=torch.float64, generator=gen)
        if snr_db is not None:
            snr = torch.full_like(snr, snr_db)
        std = torch.sqrt(sig * 10 ** (-snr / 10) / 2).float()[:, None, None, None]
        noise = torch.complex(torch.randn(hist.shape, device=sp.device, generator=gen),
                              torch.randn(hist.shape, device=sp.device, generator=gen))
        hist_n = hist + std * noise
        omega = torch.zeros(B, dtype=torch.float64, device=sp.device)
        if self.doppler == "est":
            r = (hist_n[:, :, :-1].conj() * hist_n[:, :, 1:]).to(torch.complex128).sum(dim=(1, 2, 3))
            omega = torch.angle(r)
            rh = torch.polar(torch.ones_like(omega[:, None] * self.o_h), -omega[:, None] * self.o_h).to(C64)
            rf = torch.polar(torch.ones_like(omega[:, None] * self.o_f), -omega[:, None] * self.o_f).to(C64)
            hist_n, hist, fut = hist_n * rh[:, None, :, None], hist * rh[:, None, :, None], fut * rf[:, None, :, None]
        elif self.doppler == "est_parity":
            # lag-2로 회전율(패리티 무관), 이후 마지막 프레임과 패리티가 다른 내부 프레임의 이웃 평균 대비 위상 θ
            omega = torch.angle((hist_n[:, :, :-2].conj() * hist_n[:, :, 2:]).to(torch.complex128).sum(dim=(1, 2, 3))) / 2
            Y = hist_n.to(torch.complex128) * torch.exp(-1j * omega[:, None] * self.o_h[None])[:, None, :, None]
            oi = (torch.arange(Tp, device=sp.device) - (Tp - 1))
            t = torch.arange(1, Tp - 1, device=sp.device); t = t[(oi[t] % 2) != 0]
            theta = torch.angle((((Y[:, :, t - 1] + Y[:, :, t + 1]) / 2).conj() * Y[:, :, t]).sum(dim=(1, 2, 3)))
            odd_h = (oi % 2 != 0).double(); odd_f = (torch.arange(1, Tf + 1, device=sp.device) % 2 != 0).double()
            ph_h = -omega[:, None] * self.o_h - theta[:, None] * odd_h
            ph_f = -omega[:, None] * self.o_f - theta[:, None] * odd_f
            rh = torch.polar(torch.ones_like(ph_h), ph_h).to(C64); rf = torch.polar(torch.ones_like(ph_f), ph_f).to(C64)
            hist_n, hist, fut = hist_n * rh[:, None, :, None], hist * rh[:, None, :, None], fut * rf[:, None, :, None]
        out = {"hist_n": hist_n, "fut": fut, "e": e, "w": w, "omega": omega}
        if want_clean_hist:
            out["hist"] = hist
        return out

    def iterate(self, windows_per_batch: int, shuffle: bool, gen=None, perm_gen=None, drop_last=False, **kw):
        N = len(self.index)
        order = torch.randperm(N, generator=perm_gen).to(self.sp.device) if shuffle else torch.arange(N, device=self.sp.device)
        stop = N - (N % windows_per_batch) if drop_last else N
        for s in range(0, stop, windows_per_batch):
            yield self.batch(order[s:s + windows_per_batch], gen, **kw)

    def eval_batches(self, windows_per_batch: int = 16, seed: int = 12345, **kw):
        """고정 시드·고정 순서 평가 배치 (재현 가능, 같은 실행 안에서 모든 방법이 같은 배치를 공유)."""
        g = torch.Generator(device=self.sp.device)
        g.manual_seed(seed)
        return self.iterate(windows_per_batch, False, gen=g, **kw)


def to_model_io(d: Dict, k: int):
    """배치 dict -> 모델 입력/정답 (subcarrier마다 독립 샘플).
    inp (B*n_sc, 2Tp, 16, 16) [frame0_re, frame0_im, ...], tgt (B*n_sc, 2, 16, 16) = horizon k 관측."""
    h, y = d["hist_n"], d["fut"][:, :, k - 1]
    B, S, Tp, D = h.shape
    side = int(math.isqrt(D))
    inp = torch.stack([h.real, h.imag], 3).reshape(B * S, 2 * Tp, side, side)
    tgt = torch.stack([y.real, y.imag], 2).reshape(B * S, 2, side, side)
    return inp.contiguous(), tgt.contiguous()


def from_model_out(o: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """모델 출력 (B*S, 2, 16, 16) -> complex (B, S, 256)."""
    o = o.float().reshape(B, S, 2, -1)
    return torch.complex(o[:, :, 0], o[:, :, 1])


def phase_augment(inp: torch.Tensor, tgt: torch.Tensor, gen=None):
    """공통 위상 회전 증강 (샘플별 φ~U[0,2π)): 입력 모든 프레임과 정답에 같은 e^{jφ}. 절대 위상은 정보가 아님."""
    n, C, H, W = inp.shape
    phi = 2 * math.pi * torch.rand(n, device=inp.device, generator=gen)
    c, s = torch.cos(phi)[:, None, None, None], torch.sin(phi)[:, None, None, None]

    def rot(z, F):
        z = z.view(n, F, 2, H, W)
        re, im = z[:, :, 0], z[:, :, 1]
        return torch.stack([re * c - im * s, re * s + im * c], 2).reshape(n, 2 * F, H, W)
    return rot(inp, C // 2), rot(tgt, tgt.shape[1] // 2)
