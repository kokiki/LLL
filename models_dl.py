"""
models.py -- RCG와 비교할 딥러닝 baseline 모델들 (LSTM / GRU / LSTM+CNN / Transformer)
===============================================================================
전부 data.py의 to_model_io()/from_model_out() 입출력 규약을 그대로 따른다:
  입력  x   : (N, 2*Tp, H, W)  float32   -- 채널 순서 [frame0_re, frame0_im, frame1_re, frame1_im, ...]
              (N = B*n_sc, subcarrier별 독립 샘플; H=W=16, 위성 송신 안테나 배열)
  출력  y   : (N, 2, H, W)     float32   -- [horizon k re, horizon k im]

RCG(Model7.py의 GeneratorUNet)와 동일한 입출력 텐서 규약을 쓰므로, train_dl_baseline.py 안에서
모델 이름만 바꾸면 완전히 동일한 데이터 파이프라인(causal 정규화·SNR 노이즈·Doppler 잔차 보정·
subcarrier-독립 처리)으로 학습/평가되어 RCG와 공정하게 비교된다.

여기 포함된 4종은 사용자가 프로젝트 문서(§1)에서 조사한 LLM4CP류 최신 논문들이 공통으로 쓰는
baseline 집합(RNN/LSTM/GRU/Transformer/LSTM+CNN)과 동일하다. "LSTM+CNN"은 매 타임스텝마다 CNN으로
안테나 배열의 공간(spatial) 상관을 뽑아낸 뒤 LSTM으로 시간축을 모델링하는, 해당 분야에서 가장 흔히
쓰이는 하이브리드 구조다.

설계 원칙: 전부 "평범한(vanilla)" baseline으로 유지했다 - RCG처럼 선형/비선형 경로를 분리하거나
cycle-consistency를 추가하는 등의 장치는 일부러 넣지 않았다(그런 장치를 넣기 시작하면 "baseline과
비교"가 아니라 "RCG의 변형과 비교"가 되어 버리기 때문).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 공통 유틸: 채널-인터리브 입력 (N, 2*Tp, H, W) <-> 시퀀스 (N, Tp, 2, H, W)
# ----------------------------------------------------------------------------
def _split_frames(x: torch.Tensor, Tp: int) -> torch.Tensor:
    """(N, 2*Tp, H, W) -> (N, Tp, 2, H, W). to_model_io()의 채널 순서(frame-major)와 일치."""
    N, C, H, W = x.shape
    assert C == 2 * Tp, f"입력 채널수({C}) != 2*Tp({2 * Tp})"
    return x.view(N, Tp, 2, H, W)


# ----------------------------------------------------------------------------
# 1) 순수 LSTM / GRU : 프레임을 벡터로 펼쳐 시퀀스로 처리
# ----------------------------------------------------------------------------
class RNNPredictor(nn.Module):
    """공통 구현체. cell='lstm' 또는 'gru'로 LSTMPredictor/GRUPredictor를 만듦.

    매 타임스텝의 (2,H,W) 프레임을 평탄화(2*H*W)한 뒤 입력 투영 Linear로 proj_dim까지 줄이고,
    그 시퀀스를 다층 RNN에 통과시켜 마지막 은닉상태로 horizon k 프레임을 복원한다.
    """

    def __init__(self, Tp: int, H: int = 16, W: int = 16, cell: str = "lstm",
                 proj_dim: int = 256, hidden: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.Tp, self.H, self.W = Tp, H, W
        in_dim = 2 * H * W
        self.in_proj = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True))
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(input_size=proj_dim, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        frames = _split_frames(x, self.Tp)                      # (N, Tp, 2, H, W)
        seq = frames.reshape(N, self.Tp, 2 * self.H * self.W)   # (N, Tp, 2HW)
        seq = self.in_proj(seq)                                 # (N, Tp, proj_dim)
        out, _ = self.rnn(seq)                                  # (N, Tp, hidden)
        last = out[:, -1]                                       # (N, hidden)
        y = self.decoder(last)                                  # (N, 2HW)
        return y.view(N, 2, self.H, self.W)


def LSTMPredictor(Tp: int, **kw) -> RNNPredictor:
    return RNNPredictor(Tp, cell="lstm", **kw)


def GRUPredictor(Tp: int, **kw) -> RNNPredictor:
    return RNNPredictor(Tp, cell="gru", **kw)


# ----------------------------------------------------------------------------
# 2) LSTM + CNN : 매 타임스텝 CNN으로 공간(안테나) 특징 추출 -> LSTM으로 시간축 처리 -> 디코더로 복원
# ----------------------------------------------------------------------------
class _FrameEncoder(nn.Module):
    """(2,H,W) 한 프레임 -> feat_dim 벡터. 16x16 -> 8x8 -> 4x4로 두 번 다운샘플."""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(inplace=True),   # 16->8
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(inplace=True),   # 8->4
        )
        self.proj = nn.Linear(64 * 4 * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)                        # (N, 64, 4, 4)
        z = z.flatten(1)                        # (N, 1024)
        return self.proj(z)                     # (N, feat_dim)


class _FrameDecoder(nn.Module):
    """hidden 벡터 -> (2,H,W) 프레임. 4x4 -> 8x8 -> 16x16로 두 번 업샘플."""

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 4->8
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 8->16
            nn.Conv2d(32, 2, 3, padding=1),                                             # 최종 회귀 출력
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj(h).view(-1, 64, 4, 4)
        return self.net(z)


class LSTMCNNPredictor(nn.Module):
    """매 타임스텝 CNN(공간) -> LSTM(시간) -> CNN 디코더. LLM4CP류 논문의 "LSTM+CNN" baseline과 동일 구조."""

    def __init__(self, Tp: int, H: int = 16, W: int = 16, feat_dim: int = 256,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert H == 16 and W == 16, "현재 인코더/디코더는 16x16 안테나 배열 전용(4x4까지 2회 다운샘플)"
        self.Tp, self.H, self.W = Tp, H, W
        self.encoder = _FrameEncoder(feat_dim)
        self.rnn = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = _FrameDecoder(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        frames = _split_frames(x, self.Tp)                       # (N, Tp, 2, H, W)
        flat = frames.reshape(N * self.Tp, 2, self.H, self.W)
        feats = self.encoder(flat).view(N, self.Tp, -1)          # (N, Tp, feat_dim)
        out, _ = self.rnn(feats)
        last = out[:, -1]                                        # (N, hidden)
        return self.decoder(last)                                # (N, 2, H, W)


# ----------------------------------------------------------------------------
# 3) Transformer (encoder-only, 마지막 타임스텝 토큰으로 readout)
# ----------------------------------------------------------------------------
class TransformerPredictor(nn.Module):
    def __init__(self, Tp: int, H: int = 16, W: int = 16, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.Tp, self.H, self.W = Tp, H, W
        in_dim = 2 * H * W
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(inplace=True))
        self.pos_embed = nn.Parameter(torch.zeros(1, Tp, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        frames = _split_frames(x, self.Tp)
        seq = frames.reshape(N, self.Tp, 2 * self.H * self.W)
        seq = self.in_proj(seq) + self.pos_embed
        # 과거 -> 미래 예측이므로 인과적 마스크(미래 프레임 참조 금지)는 불필요(Tp개가 전부 "과거"),
        # 다만 자기회귀적 성격을 명시하려면 causal mask를 둬도 결과는 동일(전부 입력이 history이므로).
        out = self.encoder(seq)                 # (N, Tp, d_model)
        last = out[:, -1]                        # 가장 최근 관측 토큰으로 readout
        y = self.decoder(last)
        return y.view(N, 2, self.H, self.W)


# ----------------------------------------------------------------------------
# 레지스트리
# ----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "lstm": LSTMPredictor,
    "gru": GRUPredictor,
    "lstmcnn": LSTMCNNPredictor,
    "transformer": TransformerPredictor,
}


def build_model(name: str, Tp: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 모델: {name} (사용 가능: {list(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[name](Tp, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 간단 셀프테스트: 각 모델이 (N,2*Tp,16,16) -> (N,2,16,16)을 만드는지 확인
    Tp = 12
    x = torch.randn(5, 2 * Tp, 16, 16)
    for name in MODEL_REGISTRY:
        m = build_model(name, Tp)
        y = m(x)
        assert y.shape == (5, 2, 16, 16), (name, y.shape)
        print(f"{name:12s} OK  out={tuple(y.shape)}  params={count_params(m):,}")
