"""
models.py -- 생성기 G = 선형 시간 지름길 + UNet
==========================================
UNet 부분은 기존 Model7.py의 GeneratorUNet과 그 구성요소(AdaptiveNorm2d, UNetDown, UNetUp, BottleneckMLPBlock)를
**소스 그대로** 옮긴 것 (성능이 검증된 구조 유지). 그 위에 LinearShortcutUNet을 정의:
    G(x) = W * x + UNet(x),  W: 1x1 conv (2Tp -> 2), 모든 안테나 공유 시간 필터, 0 초기화, end-to-end 학습
0 초기화라 학습 시작 시 순수 UNet과 동일. W = "마지막 프레임 복사"면 고정 잔차 연결과 동일(특수한 경우).
짧은 horizon에선 마지막 프레임 근처로, 긴 horizon에선 필요한 만큼만 학습되어 horizon 공통 구조로 사용.
"""
import torch
import torch.nn as nn

class AdaptiveNorm2d(nn.Module):
    """
    • H×W > 1 인 경우 → InstanceNorm2d (공간축 평균·분산 정규화 + γ·β)
    • H×W == 1 인 경우 → 채널축 평균·분산 정규화(LayerNorm 방식) + γ·β
      → 공간 축이 없어 InstanceNorm2d의 분산 계산이 정의되지 않으므로,
        대신 채널 축에 대해 샘플별(per-sample) 정규화를 수행해 동일한
        "입력 스케일에 적응"하는 성질을 병목 구간에서도 유지한다.

    변경 이유: 이전 버전은 H×W==1에서 고정 학습된 affine(γ·x+β)만 적용했는데,
    이는 병목 구간이 학습 시 관측한 입력 에너지 스케일에 맞춰 γ·β가 고정되어
    버려서, train/val과 다른 에너지 스케일(예: 페이딩 드리프트로 파워가 더
    낮은 구간)의 입력에 제대로 재적응하지 못하는 문제가 있었다 (test 구간에서
    예측 에너지가 실제 타겟보다 ~24% 과다 예측되는 현상으로 관측됨).
    """
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
            # 채널 축 기준 샘플별 정규화 (LayerNorm-style), 그 다음 학습된 affine
            x_flat = x.view(B, C)
            mean = x_flat.mean(dim=1, keepdim=True)
            var = x_flat.var(dim=1, keepdim=True, unbiased=False)
            x_norm = (x_flat - mean) / torch.sqrt(var + self.eps)
            x_norm = x_norm.view(B, C, 1, 1)
            g = self.gamma.view(1, C, 1, 1)
            b = self.beta .view(1, C, 1, 1)
            return x_norm * g + b
        else:
            # 평균·분산 정규화 + inst.weight·inst.bias
            return self.inst(x)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, kernel_size=4):
        super(UNetDown, self).__init__()
        # kernel=4, stride=2, padding=1: out = floor((H+2-4)/2)+1 = H/2 exactly
        # for even H -- the standard pix2pix pairing, and the EXACT inverse of
        # UNetUp's ConvTranspose2d(k=4,s=2,p=1) below (out = 2*H exactly), so
        # skip connections always match size with no adaptive_avg_pool2d
        # fallback needed. kernel_size is configurable (default 4) so this
        # class can also build the old 8-level, kernel=3 architecture for
        # ablation studies -- see GeneratorUNetOldFixed.
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False)]
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
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=4, output_padding=0):
        super(UNetUp, self).__init__()
        # kernel=4, stride=2, padding=1: out = 2*H_in exactly -- see UNetDown's
        # comment for why this (not the old k=3) is the correct exact-inverse
        # pairing. kernel_size/output_padding configurable for
        # GeneratorUNetOldFixed's ablation variant: kernel=3 (matching the
        # original architecture exactly) with output_padding=1 on ONLY its
        # up5/up6/up7 gives out=2*H_in too (verified: (H-1)*2-2+3+1=2*H),
        # fixing the old skip-size mismatch (which silently fell through to
        # adaptive_avg_pool2d, replicating the bottleneck's single value)
        # while changing NOTHING else about the original design -- not even
        # kernel_size, depth, or when spatial recovery starts (still up5).
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
        # skip_input_size kept only for call-site compatibility; with the
        # exact-inverse k=4/s=2/p=1 pairing, x always already matches
        # skip_input's spatial size, so no adaptive_avg_pool2d fallback is
        # needed (or should ever trigger -- an assert here would catch a
        # future depth/size mismatch immediately instead of silently
        # replicating values like the old try/except did).
        assert x.shape[-2:] == skip_input.shape[-2:], (
            f"UNetUp size mismatch: got {tuple(x.shape[-2:])}, "
            f"skip is {tuple(skip_input.shape[-2:])} -- depth/size no longer "
            f"line up exactly; fix the architecture rather than silently "
            f"pooling.")
        x = torch.cat((x, skip_input), 1)
        return x


class BottleneckMLPBlock(nn.Module):
    """A single channel-mixing block at the 1x1 bottleneck: Conv2d with
    kernel=1, stride=1, padding=0 has no spatial extent at all, so at 1x1
    resolution it is mathematically an ordinary per-sample linear layer over
    channels (out = W @ in + b, batched) -- i.e. an explicit, honestly-named
    MLP layer, not a conv pretending to see a neighborhood it can't have at
    a single pixel. This is what the old 8-level design's down5-down7
    actually computed (their kernel=3,stride=2,padding=1 conv is ALSO just a
    channel-mixing linear map when applied to a 1x1 input, per the same
    floor((1+2-3)/2)+1=1 formula) -- reproduced here honestly instead of
    accidentally, so the bottleneck keeps nonlinear-regression depth without
    resurrecting a fake "spatial" layer or the old up5 size-mismatch bug.
    """
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                  AdaptiveNorm2d(channels), nn.LeakyReLU(0.2)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Residual (x + f(x)), not plain f(x): a pure sequential stack of
        # N of these blocks makes gradients pass through N nonlinearities
        # with no identity shortcut -- exactly the depth-vs-optimization
        # problem ResNet's skip connections were introduced to fix. Verified
        # empirically: raising n_bottleneck_layers 3->10 (matching the old
        # architecture's true effective bottleneck-region param count, see
        # train_rcg.py's --n_bottleneck_layers help) did NOT change the
        # train-NMSE plateau at all through epoch 22, suggesting the deep
        # PLAIN stack's optimization path, not raw capacity, was the
        # bottleneck (pun intended). The residual form costs nothing extra
        # and isolates that hypothesis directly.
        return x + self.model(x)


class GeneratorUNet(nn.Module):
    # Depth reduced from 8 to 4 down/up levels: this dataset's spatial input
    # is a 16x16 antenna grid (M_ant x N_ant), not the 256x256 images the
    # original pix2pix 8-level design assumed. log2(16)=4, so 4 stride-2
    # downsamples exactly reach a 1x1 bottleneck (16->8->4->2->1); the old
    # 8-level version reached 1x1 after only 4 levels too (down4) but then
    # kept 4 MORE nominally-"downsampling" layers (down5-down8) that, given a
    # 1x1 input, mathematically cannot shrink further and just mix channels
    # -- confirmed by tracing UNetDown's conv formula: floor((1+2-3)/2)+1=1.
    # Those 4 extra layers (half the encoder, all 4 carrying Dropout(0.5))
    # were running with ZERO spatial receptive field, i.e. operating as a
    # per-pixel MLP on an already-collapsed 512-dim vector, not a U-Net.
    # Worse, the decoder's up5 tried to go from that still-1x1 state back
    # toward 2x2 using the SAME k=3,s=2,p=1 pairing, which is not an exact
    # inverse of the k=3 downsample (ConvTranspose2d out=2*H-1, not 2*H) --
    # so up5's skip concat always mismatched size and silently fell through
    # to adaptive_avg_pool2d(1x1 -> 2x2), which for a single input pixel
    # just replicates that one value into all 4 output cells. Real learned
    # spatial reconstruction only started at up6/up7 in the old design.
    #
    # Fix: exactly 4 down/up levels, using k=4,s=2,p=1 (the standard pix2pix
    # pairing) instead of k=3 -- Conv2d(k=4,s=2,p=1) halves H exactly and
    # ConvTranspose2d(k=4,s=2,p=1) doubles it exactly, so every skip
    # connection matches size with no pooling fallback ever needed (see the
    # assert in UNetUp.forward, which would now fire immediately if a future
    # change broke this instead of silently degrading to value-replication).
    # n_bottleneck_layers: explicit channel-mixing (BottleneckMLPBlock) depth
    # at the 1x1 bottleneck, restoring the nonlinear-regression capacity the
    # old design's down5-down7 provided (see BottleneckMLPBlock docstring)
    # without pretending it's spatial. Empirically needed: with this at 0,
    # train NMSE plateaus within ~5 epochs regardless of dropout (tested
    # 0.5 and 0.2, identical plateau at -8.45dB) -- a pure capacity limit,
    # not over-regularization, since lowering dropout alone did not help.
    def __init__(self, in_channels=6, out_channels=3, dropout=0.5, n_bottleneck_layers=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)   # 16 -> 8
        self.down2 = UNetDown(64, 128)                            # 8  -> 4
        self.down3 = UNetDown(128, 256, dropout=dropout)          # 4  -> 2
        self.down4 = UNetDown(256, 512, normalize=False, dropout=dropout)  # 2 -> 1 (bottleneck)

        self.bottleneck_mlp = nn.Sequential(*[
            BottleneckMLPBlock(512, dropout=dropout) for _ in range(n_bottleneck_layers)
        ])

        self.up1 = UNetUp(512, 256, dropout=dropout)   # 1 -> 2, concat d3(256) -> 512
        self.up2 = UNetUp(512, 128, dropout=dropout)   # 2 -> 4, concat d2(128) -> 256
        self.up3 = UNetUp(256, 64)                     # 4 -> 8, concat d1(64)  -> 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),   # 8 -> 16
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            # No Tanh here (unlike the original pix2pix image-generation design this
            # architecture is based on, where pixel targets are normalized to
            # [-1,1]): this dataset's clean CSI real/imag components are NOT
            # bounded to [-1,1] -- 25.8% of them exceed magnitude 1.0 (up to 3.48)
            # -- so a Tanh output would make ~1/4 of all targets structurally
            # unreachable regardless of training, concentrated on exactly the
            # highest-power (highest-NMSE-weight) blocks. Verified: this is the
            # actual bottleneck behind RCG's pooled-NMSE gap to LS/LMMSE, not the
            # loss composition or optimizer beta1 (both ablated with no effect).
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)   # 8x8,   64ch
        d2 = self.down2(d1)  # 4x4,   128ch
        d3 = self.down3(d2)  # 2x2,   256ch
        d4 = self.down4(d3)  # 1x1,   512ch (bottleneck)
        d4 = self.bottleneck_mlp(d4)  # 1x1, 512ch -- nonlinear channel-mixing depth

        u1 = self.up1(d4, d3)  # 2x2, 256+256=512ch
        u2 = self.up2(u1, d2)  # 4x4, 128+128=256ch
        u3 = self.up3(u2, d1)  # 8x8,  64+64 =128ch

        return self.final(u3)  # 16x16, out_channels


class LinearShortcutUNet(GeneratorUNet):
    """G(x) = W*x + UNet(x). 학습 가능한 선형 시간 지름길(1x1 conv, 0 초기화) + 비선형 UNet 보정."""

    def __init__(self, in_channels, out_channels=2, dropout=0.5, n_bottleneck_layers=3):
        super().__init__(in_channels, out_channels, dropout=dropout, n_bottleneck_layers=n_bottleneck_layers)
        self.lin = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        return super().forward(x) + self.lin(x)
