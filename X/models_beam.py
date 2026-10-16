"""
models_beam.py -- 배열 구조 빔 지름길 + 비선형 UNet 보정 (BSC-UNet: Beam-ShortCut UNet)
===============================================================================
models.LinearShortcutUNet(LSC-UNet)의 안테나별 선형 지름길을 **배열 구조 기반 빔 영역 선형 예측**으로 바꾼 모델.
이 파일 하나만으로 동작하도록 UNet 구성요소(AdaptiveNorm2d, UNetDown, UNetUp, BottleneckMLPBlock, GeneratorUNet)를
models.py(= Model7.py)에서 코드 그대로 옮겨 포함 (모듈 이름·구조 동일 -> 기존 체크포인트 그대로 로드됨).

  1) 빔 추정 (학습 없음, 미분 안 함): noisy history 각 프레임의 16x16 배열에 zero-pad 2D FFT(64x64) ->
     프레임 전력 합이 최대인 방향 -> 그 주변 ±1 bin을 65x65로 정밀 탐색 (각도 해상도 2π/2048) -> 평면파 조향 벡터 a
     필요한 정보는 16x16 등간격 배열이라는 구조뿐 (정답·clean 채널·궤도 정보 미사용)
  2) 빔 지름길: 빔계수 s_t = a^H h_t (t = 1..Tp) -> 학습형 복소 Tp-탭 필터 ŝ = Σ_t w_t s_t -> ŝ·a
  3) UNet 보정: 입력 = [원래 history (2Tp ch), 빔 투영 history s_t·a (2Tp ch)] -> 비선형 잔차
  출력 = UNet + 빔 지름길  (입출력 텐서 규약은 data.to_model_io와 동일: (N,2Tp,16,16) -> (N,2,16,16))
"""
import math

import torch
import torch.nn as nn


# =============================================================================
# UNet 구성요소 (models.py / Model7.py와 코드 동일)
# =============================================================================
class AdaptiveNorm2d(nn.Module):
    """H×W > 1: InstanceNorm2d (affine). H×W == 1 (병목): 채널 축 샘플별 정규화(LayerNorm 방식) + γ·β
    (공간 축이 없어 InstanceNorm 분산이 정의되지 않으므로, 입력 스케일 적응 성질을 병목에서도 유지)."""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.inst = nn.InstanceNorm2d(num_channels,
                                      affine=True,
                                      eps=eps)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta  = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        if H * W == 1:
            x_flat = x.view(B, C)
            mean = x_flat.mean(dim=1, keepdim=True)
            var = x_flat.var(dim=1, keepdim=True, unbiased=False)
            x_norm = (x_flat - mean) / torch.sqrt(var + self.eps)
            x_norm = x_norm.view(B, C, 1, 1)
            g = self.gamma.view(1, C, 1, 1)
            b = self.beta .view(1, C, 1, 1)
            return x_norm * g + b
        else:
            return self.inst(x)


class UNetDown(nn.Module):
    """Conv2d(k=4, s=2, p=1): H -> H/2 정확히 (UNetUp의 ConvTranspose2d(k=4, s=2, p=1)과 정확한 역관계)."""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_size=4, pad_mode="zeros"):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False, padding_mode=pad_mode)]
        if normalize:
            layers.append(AdaptiveNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class UNetUp(nn.Module):
    """ConvTranspose2d(k=4, s=2, p=1): H -> 2H 정확히, 이후 skip 연결과 채널 방향 concat."""
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=4, output_padding=0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size, 2, 1,
                                output_padding=output_padding, bias=False),
            AdaptiveNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input, skip_input_size=None):
        x = self.model(x)
        assert x.shape[-2:] == skip_input.shape[-2:], (
            f"UNetUp size mismatch: got {tuple(x.shape[-2:])}, "
            f"skip is {tuple(skip_input.shape[-2:])} -- depth/size no longer "
            f"line up exactly; fix the architecture rather than silently "
            f"pooling.")
        x = torch.cat((x, skip_input), 1)
        return x


class BottleneckMLPBlock(nn.Module):
    """1x1 병목의 채널 혼합 블록 (1x1 conv = 샘플별 선형층) + 잔차 연결 x + f(x)."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                  AdaptiveNorm2d(channels), nn.LeakyReLU(0.2)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.model(x)


class GeneratorUNet(nn.Module):
    """16x16 안테나 배열용 4단계 UNet: 16 -> 8 -> 4 -> 2 -> 1 (병목 MLP n_bottleneck_layers개) -> 2 -> 4 -> 8 -> 16.
    출력층에 Tanh 없음 (CSI 실수/허수 성분이 [-1, 1]로 제한되지 않음).
    pad_mode: 공간 합성곱 패딩 ('zeros' = 원래 구조, 'replicate' = 배열 가장자리 값 복제; 파라미터 구성은 동일).
    sym_out: True면 출력층을 대칭 구조(Upsample -> Conv2d k=3, padding=1)로. 원래 출력층은 ZeroPad2d((1,0,1,0)) +
             Conv2d(k=4, padding=1)라 위·왼쪽에만 0이 한 줄 더 붙어 배열 맨 윗행 오차가 커짐."""
    def __init__(self, in_channels=6, out_channels=3, dropout=0.5, n_bottleneck_layers=3, pad_mode="zeros", sym_out=False):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False, pad_mode=pad_mode)   # 16 -> 8
        self.down2 = UNetDown(64, 128, pad_mode=pad_mode)                            # 8  -> 4
        self.down3 = UNetDown(128, 256, dropout=dropout, pad_mode=pad_mode)          # 4  -> 2
        self.down4 = UNetDown(256, 512, normalize=False, dropout=dropout, pad_mode=pad_mode)  # 2 -> 1 (bottleneck)

        self.bottleneck_mlp = nn.Sequential(*[
            BottleneckMLPBlock(512, dropout=dropout) for _ in range(n_bottleneck_layers)
        ])

        self.up1 = UNetUp(512, 256, dropout=dropout)   # 1 -> 2, concat d3(256) -> 512
        self.up2 = UNetUp(512, 128, dropout=dropout)   # 2 -> 4, concat d2(128) -> 256
        self.up3 = UNetUp(256, 64)                     # 4 -> 8, concat d1(64)  -> 128

        if sym_out:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),   # 8 -> 16
                nn.Conv2d(128, out_channels, 3, padding=1, padding_mode=pad_mode),   # 대칭 패딩
            )
        else:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),   # 8 -> 16
                nn.ZeroPad2d((1, 0, 1, 0)) if pad_mode == "zeros" else nn.ReplicationPad2d((1, 0, 1, 0)),
                nn.Conv2d(128, out_channels, 4, padding=1, padding_mode=pad_mode),
            )

    def forward(self, x):
        d1 = self.down1(x)   # 8x8,   64ch
        d2 = self.down2(d1)  # 4x4,   128ch
        d3 = self.down3(d2)  # 2x2,   256ch
        d4 = self.down4(d3)  # 1x1,   512ch (bottleneck)
        d4 = self.bottleneck_mlp(d4)  # 1x1, 512ch

        u1 = self.up1(d4, d3)  # 2x2, 256+256=512ch
        u2 = self.up2(u1, d2)  # 4x4, 128+128=256ch
        u3 = self.up3(u2, d1)  # 8x8,  64+64 =128ch

        return self.final(u3)  # 16x16, out_channels


# =============================================================================
# 빔 추정 + BSC-UNet
# =============================================================================
NF = 64           # 거친 탐색 FFT 격자 (각도 해상도 2π/NF)
N_FINE = 65       # 정밀 탐색 격자 (±1 bin, 해상도 2π/(NF·(N_FINE-1)/2))


@torch.no_grad()
def manifold_beam(Hc: torch.Tensor) -> torch.Tensor:
    """Hc (N, T, 16, 16) complex noisy 관측 -> 단위 평면파 조향 벡터 a (N, 16, 16) complex.
    a[m, n] = exp(j(α m + β n)) / 16,  (α, β) = argmax Σ_t |a^H h_t|^2."""
    N, T, M, L = Hc.shape
    dev = Hc.device
    P = (torch.fft.fft2(Hc.to(torch.complex64), s=(NF, NF)).abs() ** 2).sum(1)    # (N, NF, NF)
    idx = P.reshape(N, -1).argmax(-1)
    a0 = 2 * math.pi * (idx // NF).double() / NF                                   # FFT 최대 k <-> α = 2πk/NF
    b0 = 2 * math.pi * (idx % NF).double() / NF
    m = torch.arange(M, device=dev, dtype=torch.float64)
    n = torch.arange(L, device=dev, dtype=torch.float64)
    off = torch.linspace(-1, 1, N_FINE, device=dev, dtype=torch.float64) * 2 * math.pi / NF
    ua = torch.exp(1j * ((a0[:, None] + off)[..., None] * m)).to(torch.complex64)   # (N, F, M) 위상은 double로 계산
    ub = torch.exp(1j * ((b0[:, None] + off)[..., None] * n)).to(torch.complex64)   # (N, F, L)
    Hd = Hc.to(torch.complex64)
    G = torch.einsum("nim,ntml->ntil", ua.conj(), Hd) @ ub.conj().transpose(-1, -2)[:, None]   # (N, T, F, F)
    j = (G.abs() ** 2).sum(1).reshape(N, -1).argmax(-1)
    a = a0 + off[j // N_FINE]; b = b0 + off[j % N_FINE]
    st = torch.exp(1j * (a[:, None, None] * m[:, None] + b[:, None, None] * n[None, :])) / math.sqrt(M * L)
    return st.to(Hc.dtype)


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """(N, 2F, H, W) [re0, im0, re1, im1, ...] -> (N, F, H, W) complex."""
    N, C, H, W = x.shape
    z = x.view(N, C // 2, 2, H, W)
    return torch.complex(z[:, :, 0], z[:, :, 1])


def _to_real(z: torch.Tensor) -> torch.Tensor:
    """(N, F, H, W) complex -> (N, 2F, H, W)."""
    N, F, H, W = z.shape
    return torch.stack([z.real, z.imag], 2).reshape(N, 2 * F, H, W)


class BeamShortcutUNet(nn.Module):
    """sym_out=True: UNet 출력층 대칭 구조 (state_dict에 sym_out_flag 버퍼로 표시).
    pad_mode="replicate": UNet 모든 공간 합성곱의 패딩을 가장자리 값 복제로 (replicate_flag 버퍼로 표시)."""
    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3, sym_out=False, pad_mode="zeros"):
        super().__init__()
        assert out_channels == 2
        self.Tp = in_channels // 2
        self.unet = GeneratorUNet(2 * in_channels, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers,
                                  sym_out=sym_out, pad_mode=pad_mode)
        if sym_out:
            self.register_buffer("sym_out_flag", torch.ones(1))
        if pad_mode == "replicate":
            self.register_buffer("replicate_flag", torch.ones(1))
        self.w_re = nn.Parameter(torch.zeros(self.Tp))          # 복소 Tp-탭 빔 필터 (0 초기화, LSC-UNet의 lin과 동일 방침)
        self.w_im = nn.Parameter(torch.zeros(self.Tp))

    def forward(self, x):
        Hc = _to_complex(x)                                     # (N, Tp, 16, 16)
        a = manifold_beam(Hc)                                   # (N, 16, 16)
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())          # 빔계수 history
        s_hat = s @ torch.complex(self.w_re, self.w_im)         # (N,)
        lin = _to_real((s_hat[:, None, None] * a)[:, None])     # (N, 2, 16, 16)
        proj = _to_real(s[:, :, None, None] * a[:, None])       # (N, 2Tp, 16, 16) 빔 투영(노이즈 제거) history
        return self.unet(torch.cat([x, proj], 1)) + lin


class BeamAlignedUNet(nn.Module):
    """BSC-UNet-A: BSC-UNet과 같은 빔 지름길 + **빔 정렬 좌표**의 UNet 보정 (배열 가장자리 효과 완화).
    추정 빔 u = 16·a (원소 크기 1인 평면파 위상)의 켤레를 입력에 곱하면 평면파 성분이 배열 전체에서 상수가 되어,
    합성곱의 가장자리 패딩(replicate)이 평면파 위상 기울기를 끊지 않는다. UNet은 정렬 좌표에서 잔차 R을 예측하고
    출력은 R ⊙ u 로 원래 좌표에 되돌림 (곱셈만이라 정보 손실 없음).
      입력  : [x ⊙ conj(u), s_t (빔계수 history, 배열 전체 상수)]  (2Tp + 2Tp ch)
      출력  = ŝ·a + (UNet 출력 ⊙ u)
    state_dict에 aligned_flag 버퍼를 넣어 로더가 구조를 판별."""
    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__()
        assert out_channels == 2
        self.Tp = in_channels // 2
        self.unet = GeneratorUNet(2 * in_channels, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers,
                                  pad_mode="replicate")
        self.w_re = nn.Parameter(torch.zeros(self.Tp))
        self.w_im = nn.Parameter(torch.zeros(self.Tp))
        self.register_buffer("aligned_flag", torch.ones(1))

    def forward(self, x):
        Hc = _to_complex(x)                                     # (N, Tp, 16, 16)
        a = manifold_beam(Hc)                                   # (N, 16, 16), 단위 노름
        u = a * math.sqrt(a.shape[-2] * a.shape[-1])            # 원소 크기 1
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())          # 빔계수 history
        s_hat = s @ torch.complex(self.w_re, self.w_im)
        lin = _to_real((s_hat[:, None, None] * a)[:, None])     # 빔 지름길 (BSC-UNet과 동일)
        xa = _to_real(Hc * u.conj()[:, None])                   # 빔 정렬 좌표의 history
        sa = _to_real((s / math.sqrt(a.shape[-2] * a.shape[-1]))[:, :, None, None].expand(-1, -1, *a.shape[-2:]))  # 정렬 좌표의 빔 투영 = 상수
        R = _to_complex(self.unet(torch.cat([xa, sa], 1)))       # (N, 1, 16, 16) 정렬 좌표 잔차
        return _to_real(R * u[:, None]) + lin


class BeamAlignedDirUNet(BeamAlignedUNet):
    """BSC-UNet-AD: BSC-UNet-A + 빔 방향 정보. 빔 정렬은 평면파 위상 기울기(= 위성 고각)를 입력에서 지우므로,
    추정한 기울기 α, β (rad/소자, [-π, π) -> /π 정규화)를 상수 채널 2개로 UNet에 추가해 되돌려 준다."""
    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__(in_channels, out_channels, dropout, n_bottleneck_layers)
        self.unet = GeneratorUNet(2 * in_channels + 2, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers,
                                  pad_mode="replicate")
        self.register_buffer("aligned_dir_flag", torch.ones(1))

    def forward(self, x):
        Hc = _to_complex(x)
        a = manifold_beam(Hc)
        M, L = a.shape[-2:]
        u = a * math.sqrt(M * L)
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())
        s_hat = s @ torch.complex(self.w_re, self.w_im)
        lin = _to_real((s_hat[:, None, None] * a)[:, None])
        xa = _to_real(Hc * u.conj()[:, None])
        sa = _to_real((s / math.sqrt(M * L))[:, :, None, None].expand(-1, -1, M, L))
        alpha = torch.angle(u[:, 1, 0] * u[:, 0, 0].conj()) / math.pi            # 행 방향 위상 기울기 (고각)
        beta = torch.angle(u[:, 0, 1] * u[:, 0, 0].conj()) / math.pi             # 열 방향 위상 기울기
        dirs = torch.stack([alpha, beta], 1)[:, :, None, None].expand(-1, -1, M, L).to(x.dtype)
        R = _to_complex(self.unet(torch.cat([xa, sa, dirs], 1)))
        return _to_real(R * u[:, None]) + lin


def build_from_state_dict(sd, Tp, dropout=0.5):
    """체크포인트 키로 BSC 계열 구조 판별: aligned_poly_flag -> AP, aligned_dir_flag -> AD, aligned_flag -> A, 그 외 w_re -> BSC-UNet."""
    if "aligned_poly_flag" in sd:
        from models_beam_poly import BeamAlignedPolyUNet          # BSC-UNet-AP (models_beam_poly.py)
        return BeamAlignedPolyUNet(2 * Tp, int(sd["poly_k"].item()), 2, dropout=dropout)
    if "aligned_dir_flag" in sd:
        return BeamAlignedDirUNet(2 * Tp, 2, dropout=dropout)
    if "aligned_flag" in sd:
        return BeamAlignedUNet(2 * Tp, 2, dropout=dropout)
    return BeamShortcutUNet(2 * Tp, 2, dropout=dropout, sym_out="sym_out_flag" in sd,
                            pad_mode="replicate" if "replicate_flag" in sd else "zeros")
