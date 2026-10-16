"""
models_beam_poly.py -- BSC-UNet-AP: BSC-UNet-A(빔 정렬 입력) + 물리 기반 빔계수 지름길 (Poly2 외삽)
  긴 horizon에서는 Doppler 변화율 잔차가 만드는 2차 위상이 지배적이라, 빔계수 s(t)의 위상 2차 / 진폭 1차
  최소제곱 외삽(Poly2_blind baseline과 같은 식, 학습 없음)을 빔 지름길에 더하고 UNet 입력에도 넣는다.
    s      : 배열 매니폴드 빔 a에 투영한 빔계수 history (Tp)            -- BSC-UNet-A와 동일
    s_poly : s의 위상 2차 / 진폭 1차 외삽값 (t = k)                       -- 윈도우마다 history만으로 계산
    지름길 = (s · w + c · s_poly) · a     (w: 학습 Tp탭 필터 초기 0, c: 학습 복소 이득 초기 1 -> 시작점 = Poly2 외삽)
    UNet 입력 = [x ⊙ conj(u), s(정렬 좌표 상수), s_poly(정렬 좌표 상수)],  출력 = 지름길 + UNet ⊙ u
  horizon k(스텝)는 구조의 일부라 poly_k 버퍼로 저장 (로더가 같은 k로 재구성).
"""
import math

import torch
import torch.nn as nn

from models_beam import BeamAlignedUNet, GeneratorUNet, _to_complex, _to_real, manifold_beam


def poly2_extrapolate(s: torch.Tensor, k: int) -> torch.Tensor:
    """s (N, Tp) 복소 빔계수 -> t = k 외삽값 (N,). 위상: 인접 차분 wrap 후 누적(unwrap) -> 2차 LS, 진폭: 1차 LS (>= 0).
    baselines.py Poly2_blind와 같은 식 (시간축 t = -(Tp-1)..0, 예측 시점 t = k)."""
    Tp = s.shape[-1]
    s64 = s.to(torch.complex128)
    t = torch.arange(Tp, device=s.device, dtype=torch.float64) - (Tp - 1)
    tk = torch.tensor([float(k)], device=s.device, dtype=torch.float64)
    ang = torch.angle(s64)
    dd = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, 2 * math.pi) - math.pi
    ph = torch.cat([ang[:, :1], ang[:, :1] + torch.cumsum(dd, -1)], -1)
    V = lambda x, deg: torch.stack([x ** i for i in range(deg, -1, -1)], -1)
    amp = torch.clamp((s64.abs() @ torch.linalg.pinv(V(t, 1)).T) @ V(tk, 1).T, min=0)[:, 0]
    phk = ((ph @ torch.linalg.pinv(V(t, 2)).T) @ V(tk, 2).T)[:, 0]
    return (amp * torch.exp(1j * phk)).to(s.dtype)


class BeamAlignedPolyUNet(BeamAlignedUNet):
    """BSC-UNet-AP. state_dict에 aligned_poly_flag, poly_k 버퍼 (build_from_state_dict가 판별)."""
    def __init__(self, in_channels, k_steps, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__(in_channels, out_channels, dropout, n_bottleneck_layers)
        self.unet = GeneratorUNet(2 * in_channels + 2, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers,
                                  pad_mode="replicate")
        self.c_re = nn.Parameter(torch.ones(1))
        self.c_im = nn.Parameter(torch.zeros(1))
        self.register_buffer("aligned_poly_flag", torch.ones(1))
        self.register_buffer("poly_k", torch.tensor([float(k_steps)]))

    def forward(self, x):
        Hc = _to_complex(x)                                         # (N, Tp, 16, 16)
        a = manifold_beam(Hc)
        M, L = a.shape[-2:]
        u = a * math.sqrt(M * L)
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())               # 빔계수 history
        s_poly = poly2_extrapolate(s.detach(), int(self.poly_k.item()))   # 학습 파라미터 없는 물리 외삽
        s_hat = s @ torch.complex(self.w_re, self.w_im) + torch.complex(self.c_re, self.c_im) * s_poly
        lin = _to_real((s_hat[:, None, None] * a)[:, None])
        xa = _to_real(Hc * u.conj()[:, None])
        sa = _to_real((s / math.sqrt(M * L))[:, :, None, None].expand(-1, -1, M, L))
        sp = _to_real((s_poly / math.sqrt(M * L))[:, None, None, None].expand(-1, 1, M, L))
        R = _to_complex(self.unet(torch.cat([xa, sa, sp], 1)))
        return _to_real(R * u[:, None]) + lin
