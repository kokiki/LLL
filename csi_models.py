"""
csi_models.py -- LEO NTN CSI 예측 모델 (models.py + models_beam.py 통합)
===============================================================================
입출력 규약 (data.to_model_io와 동일): x (N, 2Tp, 16, 16) [frame0_re, frame0_im, ...] -> (N, 2, 16, 16) horizon k 예측

  GeneratorUNet       : 16x16 안테나 배열용 4단계 UNet (지름길 없는 ablation, 다른 모델의 보정부)
  LinearShortcutUNet  : LSC-UNet = 안테나 공유 선형 시간 필터(1x1 conv) + UNet 보정 (ablation)
  BeamShortcutUNet    : BSC-UNet = 배열 구조 빔 지름길 + UNet 보정 (제안 모델; 개선 학습 설정으로 학습한 것이 BSC-UNet-I)
      1) 빔 추정 (학습 없음): noisy history에 zero-pad 2D FFT(64x64) -> 최대 전력 방향 주변 65x65 정밀 탐색
         -> 평면파 조향 벡터 a (필요한 정보는 16x16 등간격 배열 구조뿐)
      2) 빔 지름길: 빔계수 s_t = a^H h_t -> 학습형 복소 Tp-탭 필터 ŝ = Σ_t w_t s_t -> ŝ·a
      3) UNet 보정: 입력 [원래 history, 빔 투영 history s_t·a] -> 비선형 잔차.  출력 = UNet + ŝ·a
      옵션 (가장자리 효과 ablation): sym_out (대칭 출력층), pad_mode="replicate"
  BeamAlignedUNet     : BSC-UNet-A = 빔 정렬 좌표 UNet + replicate 패딩 (v10 방위각 포함 데이터에서 BSC-UNet-AI로 학습)
  build_model_from_state_dict : 체크포인트 키로 위 구조를 판별해 생성

레이어 이름·순서는 기존 Model7.py / models.py / models_beam.py와 동일 -> 기존 체크포인트 그대로 로드.
"""
import math

import torch
import torch.nn as nn


# =============================================================================
# UNet 구성요소
# =============================================================================
class AdaptiveNorm2d(nn.Module):
    """H×W > 1: InstanceNorm2d (affine). H×W == 1 (병목): 채널 축 샘플별 정규화(LayerNorm 방식) + γ·β."""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.inst = nn.InstanceNorm2d(num_channels, affine=True, eps=eps)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        if H * W == 1:
            x_flat = x.view(B, C)
            mean = x_flat.mean(dim=1, keepdim=True)
            var = x_flat.var(dim=1, keepdim=True, unbiased=False)
            x_norm = ((x_flat - mean) / torch.sqrt(var + self.eps)).view(B, C, 1, 1)
            return x_norm * self.gamma.view(1, C, 1, 1) + self.beta.view(1, C, 1, 1)
        return self.inst(x)


class UNetDown(nn.Module):
    """Conv2d(k=4, s=2, p=1): H -> H/2 (UNetUp의 ConvTranspose2d(k=4, s=2, p=1)과 정확한 역관계)."""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, pad_mode="zeros"):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False, padding_mode=pad_mode)]
        if normalize:
            layers.append(AdaptiveNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ConvTranspose2d(k=4, s=2, p=1): H -> 2H, 이후 skip 연결과 채널 방향 concat."""
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False), AdaptiveNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        assert x.shape[-2:] == skip_input.shape[-2:], f"UNetUp size mismatch: {tuple(x.shape[-2:])} vs {tuple(skip_input.shape[-2:])}"
        return torch.cat((x, skip_input), 1)


class BottleneckMLPBlock(nn.Module):
    """1x1 병목의 채널 혼합 블록 (1x1 conv = 샘플별 선형층) + 잔차 연결 x + f(x)."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, kernel_size=1, bias=False), AdaptiveNorm2d(channels), nn.LeakyReLU(0.2)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.model(x)


class GeneratorUNet(nn.Module):
    """16x16 배열용 4단계 UNet: 16 -> 8 -> 4 -> 2 -> 1 (병목 MLP) -> 2 -> 4 -> 8 -> 16. 출력층 Tanh 없음.
    pad_mode: 공간 합성곱 패딩 ('zeros' 기본, 'replicate').  sym_out: 출력층을 대칭(Conv2d k=3, padding=1)으로."""
    def __init__(self, in_channels=6, out_channels=3, dropout=0.5, n_bottleneck_layers=3, pad_mode="zeros", sym_out=False):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False, pad_mode=pad_mode)          # 16 -> 8
        self.down2 = UNetDown(64, 128, pad_mode=pad_mode)                                   # 8  -> 4
        self.down3 = UNetDown(128, 256, dropout=dropout, pad_mode=pad_mode)                 # 4  -> 2
        self.down4 = UNetDown(256, 512, normalize=False, dropout=dropout, pad_mode=pad_mode)  # 2 -> 1
        self.bottleneck_mlp = nn.Sequential(*[BottleneckMLPBlock(512, dropout=dropout) for _ in range(n_bottleneck_layers)])
        self.up1 = UNetUp(512, 256, dropout=dropout)   # 1 -> 2
        self.up2 = UNetUp(512, 128, dropout=dropout)   # 2 -> 4
        self.up3 = UNetUp(256, 64)                     # 4 -> 8
        if sym_out:
            self.final = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(128, out_channels, 3, padding=1, padding_mode=pad_mode))
        else:
            self.final = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.ZeroPad2d((1, 0, 1, 0)) if pad_mode == "zeros" else nn.ReplicationPad2d((1, 0, 1, 0)),
                                       nn.Conv2d(128, out_channels, 4, padding=1, padding_mode=pad_mode))

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.bottleneck_mlp(self.down4(d3))
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


# =============================================================================
# LSC-UNet (ablation)
# =============================================================================
class LinearShortcutUNet(GeneratorUNet):
    """G(x) = W*x + UNet(x). W: 1x1 conv (2Tp -> 2), 모든 안테나 공유 시간 필터, 0 초기화."""
    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__(in_channels, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers)
        self.lin = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        return super().forward(x) + self.lin(x)


# =============================================================================
# 빔 추정 + BSC 계열
# =============================================================================
NF = 64           # 거친 탐색 FFT 격자 (각도 해상도 2π/NF)
N_FINE = 65       # 정밀 탐색 격자 (±1 bin)


@torch.no_grad()
def manifold_beam(Hc: torch.Tensor) -> torch.Tensor:
    """Hc (N, T, 16, 16) complex noisy 관측 -> 단위 평면파 조향 벡터 a (N, 16, 16).
    a[m, n] = exp(j(α m + β n)) / 16,  (α, β) = argmax Σ_t |a^H h_t|^2."""
    N, T, M, L = Hc.shape
    dev = Hc.device
    P = (torch.fft.fft2(Hc.to(torch.complex64), s=(NF, NF)).abs() ** 2).sum(1)
    idx = P.reshape(N, -1).argmax(-1)
    a0 = 2 * math.pi * (idx // NF).double() / NF
    b0 = 2 * math.pi * (idx % NF).double() / NF
    m = torch.arange(M, device=dev, dtype=torch.float64)
    n = torch.arange(L, device=dev, dtype=torch.float64)
    off = torch.linspace(-1, 1, N_FINE, device=dev, dtype=torch.float64) * 2 * math.pi / NF
    ua = torch.exp(1j * ((a0[:, None] + off)[..., None] * m)).to(torch.complex64)
    ub = torch.exp(1j * ((b0[:, None] + off)[..., None] * n)).to(torch.complex64)
    G = torch.einsum("nim,ntml->ntil", ua.conj(), Hc.to(torch.complex64)) @ ub.conj().transpose(-1, -2)[:, None]
    j = (G.abs() ** 2).sum(1).reshape(N, -1).argmax(-1)
    a = a0 + off[j // N_FINE]; b = b0 + off[j % N_FINE]
    st = torch.exp(1j * (a[:, None, None] * m[:, None] + b[:, None, None] * n[None, :])) / math.sqrt(M * L)
    return st.to(Hc.dtype)


def _to_complex(x):
    """(N, 2F, H, W) [re0, im0, re1, im1, ...] -> (N, F, H, W) complex."""
    N, C, H, W = x.shape
    z = x.view(N, C // 2, 2, H, W)
    return torch.complex(z[:, :, 0], z[:, :, 1])


def _to_real(z):
    """(N, F, H, W) complex -> (N, 2F, H, W)."""
    N, F, H, W = z.shape
    return torch.stack([z.real, z.imag], 2).reshape(N, 2 * F, H, W)


def _beam_shortcut(model, Hc):
    """공통: 빔 a, 빔계수 history s, 빔 지름길 출력 lin (N, 2, 16, 16)."""
    a = manifold_beam(Hc)
    s = torch.einsum("ntij,nij->nt", Hc, a.conj())
    s_hat = s @ torch.complex(model.w_re, model.w_im)
    return a, s, _to_real((s_hat[:, None, None] * a)[:, None])


class BeamShortcutUNet(nn.Module):
    """BSC-UNet (제안 모델). sym_out / pad_mode="replicate"는 ablation 옵션 (각각 sym_out_flag / replicate_flag 버퍼로 표시)."""
    extra_in = 0                                              # 하위 클래스가 UNet 입력 채널을 추가할 때 사용

    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3, sym_out=False, pad_mode="zeros"):
        super().__init__()
        assert out_channels == 2
        self.Tp = in_channels // 2
        self.unet = GeneratorUNet(2 * in_channels + self.extra_in, out_channels, dropout=dropout,
                                  n_bottleneck_layers=n_bottleneck_layers, sym_out=sym_out, pad_mode=pad_mode)
        if sym_out:
            self.register_buffer("sym_out_flag", torch.ones(1))
        if pad_mode == "replicate":
            self.register_buffer("replicate_flag", torch.ones(1))
        self.w_re = nn.Parameter(torch.zeros(self.Tp))       # 복소 Tp-탭 빔 필터 (0 초기화)
        self.w_im = nn.Parameter(torch.zeros(self.Tp))

    def forward(self, x):
        Hc = _to_complex(x)
        a, s, lin = _beam_shortcut(self, Hc)
        proj = _to_real(s[:, :, None, None] * a[:, None])    # 빔 투영(노이즈 제거) history
        return self.unet(torch.cat([x, proj], 1)) + lin


_POLY2_CACHE = {}


def _poly2_mats(Tp, k, device):
    """최소제곱 의사역행렬과 외삽 시점 Vandermonde 행 (Tp, k, device별 상수, 한 번만 계산).
    추론 때마다 pinv(SVD)를 다시 계산하지 않으므로 더 빠르고 CUDA Graph 캡처도 가능하다 (결과는 동일)."""
    key = (Tp, k, str(device))
    if key not in _POLY2_CACHE:
        t = torch.arange(Tp, device=device, dtype=torch.float64) - (Tp - 1)
        tk = torch.tensor([float(k)], device=device, dtype=torch.float64)
        V = lambda x, deg: torch.stack([x ** i for i in range(deg, -1, -1)], -1)
        _POLY2_CACHE[key] = (torch.linalg.pinv(V(t, 1)).T, V(tk, 1).T, torch.linalg.pinv(V(t, 2)).T, V(tk, 2).T)
    return _POLY2_CACHE[key]


def poly2_extrapolate(s: torch.Tensor, k: int) -> torch.Tensor:
    """s (N, Tp) 복소 빔계수 history -> t = k 외삽값 (N,). 위상: 인접 차분 wrap 후 누적(unwrap) -> 2차 최소제곱,
    진폭: 1차 최소제곱 (>= 0). baselines.py Poly2_blind / ManifoldPoly2와 같은 식 (시간축 o = -(Tp-1)..0, 예측 o = k)."""
    Tp = s.shape[-1]
    s64 = s.to(torch.complex128)
    P1T, V1k, P2T, V2k = _poly2_mats(Tp, k, s.device)
    ang = torch.angle(s64)
    dd = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, 2 * math.pi) - math.pi
    ph = torch.cat([ang[:, :1], ang[:, :1] + torch.cumsum(dd, -1)], -1)
    amp = torch.clamp((s64.abs() @ P1T) @ V1k, min=0)[:, 0]
    phk = ((ph @ P2T) @ V2k)[:, 0]
    return (amp * torch.exp(1j * phk)).to(s.dtype)


class BeamShortcutPolyUNet(BeamShortcutUNet):
    """BSC-UNet-P: BSC-UNet + 2차 위상 외삽 빔 지름길. Doppler 변화율 잔차가 만드는 에피소드별 2차 위상은 고정 선형 필터로
    외삽할 수 없으므로, 빔계수 history s의 위상 2차/진폭 1차 외삽값 s_poly(k)(학습 없음)를 지름길에 더한다.
      지름길 ŝ = Σ_l w_l s(l) + c·s_poly   (w: 0 초기화, c: 복소 이득 1 초기화 -> 시작점 = ManifoldPoly2)
      UNet 입력 = [x, s·a (Tp 프레임), s_poly·a (1 프레임)],  출력 = UNet + ŝ·a
    horizon k(스텝)는 구조의 일부라 poly_k 버퍼로 저장. poly_flag 버퍼로 표시."""
    extra_in = 2

    def __init__(self, in_channels, k_steps, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__(in_channels, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers)
        self.c_re = nn.Parameter(torch.ones(1))
        self.c_im = nn.Parameter(torch.zeros(1))
        self.register_buffer("poly_flag", torch.ones(1))
        self.register_buffer("poly_k", torch.tensor([float(k_steps)]))
        self.k_steps = int(k_steps)                           # forward에서 .item() 동기화 없이 쓰기 위한 파이썬 정수 (CUDA Graph 가능)

    def forward(self, x):
        Hc = _to_complex(x)
        a = manifold_beam(Hc)
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())
        s_poly = poly2_extrapolate(s.detach(), self.k_steps)
        s_hat = s @ torch.complex(self.w_re, self.w_im) + torch.complex(self.c_re, self.c_im) * s_poly
        lin = _to_real((s_hat[:, None, None] * a)[:, None])
        proj = _to_real(s[:, :, None, None] * a[:, None])
        proj_poly = _to_real((s_poly[:, None, None] * a)[:, None])
        return self.unet(torch.cat([x, proj, proj_poly], 1)) + lin


class BeamAlignedUNet(nn.Module):
    """BSC-UNet-A (ablation): 빔 정렬 좌표 UNet. 입력에 conj(u) (u = 16·a)를 곱해 평면파 위상 기울기를 제거하고
    replicate 패딩 UNet이 정렬 좌표 잔차 R을 예측, 출력 = ŝ·a + R ⊙ u. aligned_flag 버퍼로 표시."""
    extra_in = 0                                              # 하위 클래스가 UNet 입력 채널을 추가할 때 사용

    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__()
        assert out_channels == 2
        self.Tp = in_channels // 2
        self.unet = GeneratorUNet(2 * in_channels + self.extra_in, out_channels, dropout=dropout,
                                  n_bottleneck_layers=n_bottleneck_layers, pad_mode="replicate")
        self.w_re = nn.Parameter(torch.zeros(self.Tp))
        self.w_im = nn.Parameter(torch.zeros(self.Tp))
        self.register_buffer("aligned_flag", torch.ones(1))

    def forward(self, x):
        Hc = _to_complex(x)
        a, s, lin = _beam_shortcut(self, Hc)
        M, L = a.shape[-2:]
        u = a * math.sqrt(M * L)
        xa = _to_real(Hc * u.conj()[:, None])                                      # 빔 정렬 좌표 history
        sa = _to_real((s / math.sqrt(M * L))[:, :, None, None].expand(-1, -1, M, L))  # 정렬 좌표 빔 투영 (상수)
        R = _to_complex(self.unet(torch.cat([xa, sa], 1)))
        return _to_real(R * u[:, None]) + lin


class BeamAlignedPolyUNet(BeamAlignedUNet):
    """BSC-UNet-AP: 빔 정렬 좌표 UNet (BSC-UNet-A) + 2차 위상 외삽 빔 지름길 (BSC-UNet-P와 같은 지름길).
    방위각·off-nadir가 다양한 기하(v10)에서도 UNet이 방향과 무관한 정렬 좌표 잔차만 보도록 하고, Doppler 변화율의 2차 위상은
    지름길의 외삽으로 따라간다.
      지름길 ŝ = Σ_l w_l s(l) + c·s_p(k)   (w: 0 초기화, c: 1 초기화 -> 시작점 = ManifoldPoly2)
      UNet 입력 = [x ⊙ conj(u), s/16 (Tp, 배열 상수), s_p/16 (1, 배열 상수)],  출력 = ŝ·a + UNet ⊙ u
    aligned_poly_flag, poly_k 버퍼로 표시 (v10 시험 학습의 체크포인트 형식과 같음)."""
    extra_in = 2

    def __init__(self, in_channels, k_steps, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__(in_channels, out_channels, dropout, n_bottleneck_layers)
        self.c_re = nn.Parameter(torch.ones(1))
        self.c_im = nn.Parameter(torch.zeros(1))
        self.register_buffer("aligned_poly_flag", torch.ones(1))
        self.register_buffer("poly_k", torch.tensor([float(k_steps)]))
        self.k_steps = int(k_steps)

    def forward(self, x):
        Hc = _to_complex(x)
        a = manifold_beam(Hc)
        M, L = a.shape[-2:]
        u = a * math.sqrt(M * L)
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())
        s_poly = poly2_extrapolate(s.detach(), self.k_steps)
        s_hat = s @ torch.complex(self.w_re, self.w_im) + torch.complex(self.c_re, self.c_im) * s_poly
        lin = _to_real((s_hat[:, None, None] * a)[:, None])
        xa = _to_real(Hc * u.conj()[:, None])
        sa = _to_real((s / math.sqrt(M * L))[:, :, None, None].expand(-1, -1, M, L))
        sp = _to_real((s_poly / math.sqrt(M * L))[:, None, None, None].expand(-1, 1, M, L))
        R = _to_complex(self.unet(torch.cat([xa, sa, sp], 1)))
        return _to_real(R * u[:, None]) + lin


def build_model_from_state_dict(sd, Tp, dropout=0.5):
    """체크포인트 키로 구조 판별 -> 생성 (가중치는 호출 측에서 load_state_dict)."""
    if "aligned_poly_flag" in sd:
        return BeamAlignedPolyUNet(2 * Tp, int(sd["poly_k"].item()), 2, dropout=dropout)
    if "aligned_dir_flag" in sd:
        raise ValueError("BSC-UNet-AD 체크포인트는 csi_models.py에서 지원하지 않음 (논문 제외 시험 모델)")
    if "poly_flag" in sd:
        return BeamShortcutPolyUNet(2 * Tp, int(sd["poly_k"].item()), 2, dropout=dropout)
    if "aligned_flag" in sd:
        return BeamAlignedUNet(2 * Tp, 2, dropout=dropout)
    if "w_re" in sd:
        return BeamShortcutUNet(2 * Tp, 2, dropout=dropout, sym_out="sym_out_flag" in sd,
                                pad_mode="replicate" if "replicate_flag" in sd else "zeros")
    if "lin.weight" in sd:
        return LinearShortcutUNet(2 * Tp, 2, dropout=dropout)
    return GeneratorUNet(2 * Tp, 2, dropout=dropout)
