"""
bench_inference.py -- BSC-UNet-I 추론 최적화 시험: 추론 시간과 NMSE(정확도 유지 여부)를 함께 측정
  변형: fp32 (기준) / fp16 UNet (빔 추정·지름길은 fp32, 복소 FFT 때문) / CUDA Graph (호출 오버헤드 제거) /
        빔 정밀 탐색 격자 축소 (65 -> 33, 17점)
  NMSE: v9b test 전체 윈도우 (evaluate.py와 같은 고정 시드 배치, pooled), k = 1, 16
  시간: 배치 12 (윈도우 1개 = 부반송파 12개), 300; 예열 후 반복 평균
출력: runs/bench_inference.json
"""
import json
import math
import time

import numpy as np
import torch

import csi_models as cm
from data import Split, Windows, to_model_io

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
KS = [1, 16]
TP = 20


def make_beam(n_fine):
    """manifold_beam과 같되 정밀 탐색 점 수만 바꾼 버전 (n_fine = 65이면 원본과 동일)."""
    @torch.no_grad()
    def beam(Hc):
        N, T, M, L = Hc.shape
        dev = Hc.device
        P = (torch.fft.fft2(Hc.to(torch.complex64), s=(cm.NF, cm.NF)).abs() ** 2).sum(1)
        idx = P.reshape(N, -1).argmax(-1)
        a0 = 2 * math.pi * (idx // cm.NF).double() / cm.NF
        b0 = 2 * math.pi * (idx % cm.NF).double() / cm.NF
        m = torch.arange(M, device=dev, dtype=torch.float64)
        n = torch.arange(L, device=dev, dtype=torch.float64)
        off = torch.linspace(-1, 1, n_fine, device=dev, dtype=torch.float64) * 2 * math.pi / cm.NF
        ua = torch.exp(1j * ((a0[:, None] + off)[..., None] * m)).to(torch.complex64)
        ub = torch.exp(1j * ((b0[:, None] + off)[..., None] * n)).to(torch.complex64)
        G = torch.einsum("nim,ntml->ntil", ua.conj(), Hc.to(torch.complex64)) @ ub.conj().transpose(-1, -2)[:, None]
        j = (G.abs() ** 2).sum(1).reshape(N, -1).argmax(-1)
        a = a0 + off[j // n_fine]; b = b0 + off[j % n_fine]
        st = torch.exp(1j * (a[:, None, None] * m[:, None] + b[:, None, None] * n[None, :])) / math.sqrt(M * L)
        return st.to(Hc.dtype)
    return beam


def make_forward(G, half=False, n_fine=65):
    """BeamShortcutUNet.forward와 같은 계산. half=True면 UNet만 fp16 (출력은 fp32로 되돌림)."""
    beam = make_beam(n_fine)
    unet = G.unet.half() if half else G.unet
    w = torch.complex(G.w_re, G.w_im)

    @torch.no_grad()
    def fwd(x):
        Hc = cm._to_complex(x)
        a = beam(Hc)
        s = torch.einsum("ntij,nij->nt", Hc, a.conj())
        lin = cm._to_real(((s @ w)[:, None, None] * a)[:, None])
        proj = cm._to_real(s[:, :, None, None] * a[:, None])
        inp = torch.cat([x, proj], 1)
        out = unet(inp.half()).float() if half else unet(inp)
        return out + lin
    return fwd


def graphed(fwd, N):
    """정적 입력 크기 N의 CUDA Graph로 캡처 -> 호출은 입력 복사 + replay."""
    x_static = torch.randn(N, 2 * TP, 16, 16, device="cuda")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd(x_static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = fwd(x_static)

    def run(x):
        x_static.copy_(x); g.replay(); return y_static
    return run


def load(k, half=False):
    sd = torch.load(f"runs/v9b_Tp20_bscI_k{k}ms/best_G.pt", map_location="cuda", weights_only=True)
    G = cm.build_model_from_state_dict(sd, TP); G.load_state_dict(sd)
    return G.cuda().eval()


def timeit(f, x, reps):
    for _ in range(10):
        f(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        f(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


VARIANTS = {"fp32 (기준)": dict(half=False, n_fine=65, graph=False),
            "fp16 UNet": dict(half=True, n_fine=65, graph=False),
            "fp32 + CUDA Graph": dict(half=False, n_fine=65, graph=True),
            "fp16 + CUDA Graph": dict(half=True, n_fine=65, graph=True),
            "fp16 + Graph + 정밀탐색 33점": dict(half=True, n_fine=33, graph=True),
            "fp16 + Graph + 정밀탐색 17점": dict(half=True, n_fine=17, graph=True)}


def main():
    torch.backends.cudnn.benchmark = True
    te = Split(D, "test")
    res = {}
    for name, cfg in VARIANTS.items():
        r = {"nmse_db": {}, "ms": {}}
        for k in KS:                                     # 정확도: test 전체 (배치 크기 = 16 윈도우 x 12 = 192)
            fwd = make_forward(load(k), cfg["half"], cfg["n_fine"])
            W = Windows(te, TP, 16)
            num = den = 0.0
            f = None
            for d in W.eval_batches(16):
                x, y = to_model_io(d, k)
                if cfg["graph"]:
                    if f is None or x.shape[0] != f[1]:
                        f = (graphed(fwd, x.shape[0]), x.shape[0])
                    p = f[0](x)
                else:
                    p = fwd(x)
                num += torch.sum((p - y) ** 2).item(); den += torch.sum(y ** 2).item()
            r["nmse_db"][str(k)] = 10 * np.log10(num / den)
        fwd = make_forward(load(16), cfg["half"], cfg["n_fine"])
        for N, reps in ((12, 500), (300, 100)):         # 시간
            x = torch.randn(N, 2 * TP, 16, 16, device="cuda")
            f = graphed(fwd, N) if cfg["graph"] else fwd
            r["ms"][str(N)] = timeit(f, x, reps)
        res[name] = r
        print(f"{name:28s} NMSE k=1 {r['nmse_db']['1']:8.3f} dB, k=16 {r['nmse_db']['16']:8.3f} dB | "
              f"배치12 {r['ms']['12']:6.2f} ms, 배치300 {r['ms']['300']:6.2f} ms", flush=True)
    base = res["fp32 (기준)"]
    print("\n기준 대비: ΔNMSE [dB] (양수 = 나빠짐), 속도 배수")
    for name, r in res.items():
        print(f"{name:28s} ΔNMSE k=1 {r['nmse_db']['1'] - base['nmse_db']['1']:+.4f}, k=16 {r['nmse_db']['16'] - base['nmse_db']['16']:+.4f} | "
              f"배치12 x{base['ms']['12'] / r['ms']['12']:.2f}, 배치300 x{base['ms']['300'] / r['ms']['300']:.2f}")
    json.dump({"gpu": torch.cuda.get_device_name(0), "results": res}, open("runs/bench_inference.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
