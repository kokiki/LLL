"""
bench_inference_p.py -- BSC-UNet-P(최종 모델) 추론 시간과 최적화 시 NMSE 유지 여부 (BSC-UNet-I와 같은 조건에서 비교)
  변형: fp32 (기준) / fp16 UNet (빔 추정·지름길·2차 외삽은 fp32) / CUDA Graph / fp16 + CUDA Graph
  시간: 배치 12 (윈도우 1개 = 부반송파 12개), 300 (단말 25명분); 예열 10회 후 반복 평균
  NMSE: v9b κ = 1 test 전체 (evaluate.py와 같은 고정 시드 배치, pooled), k = 1, 16
  출력: runs/bench_inference_p.json
  주의: 다른 GPU 작업과 동시에 측정하면 절대 시간이 늘어난다 (같은 조건에서 측정한 BSC-UNet-I와의 상대 비교는 유효).
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn

import csi_models as cm
from data import Split, Windows, rescale_doppler_rate, to_model_io

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"
TP = 20
CKPT = {"BSC-UNet-P": "runs/v9b_Tp20_bscP_kappa1_k{k}ms/best_G.pt", "BSC-UNet-I": "runs/v9b_Tp20_bscI_kappa1_k{k}ms/best_G.pt"}


class HalfUNet(nn.Module):
    """UNet만 fp16으로 계산하고 출력은 fp32로 되돌림."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet.half()

    def forward(self, x):
        return self.unet(x.half()).float()


def load(path, half):
    sd = torch.load(path, map_location="cuda", weights_only=True)
    G = cm.build_model_from_state_dict(sd, TP); G.load_state_dict(sd); G = G.cuda().eval()
    if half:
        G.unet = HalfUNet(G.unet)
    return G


def graphed(G, N):
    x_static = torch.randn(N, 2 * TP, 16, 16, device="cuda")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(3):
            G(x_static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        y_static = G(x_static)

    def run(x):
        x_static.copy_(x); g.replay(); return y_static
    return run


@torch.no_grad()
def timeit(f, x, reps):
    for _ in range(10):
        f(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        f(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def main():
    torch.backends.cudnn.benchmark = True
    te = Split(D, "test"); rescale_doppler_rate(te, 1.0)
    res = {"gpu": torch.cuda.get_device_name(0)}
    for model, tmpl in CKPT.items():
        res[model] = {}
        for vname, half, graph in (("fp32", False, False), ("fp16 UNet", True, False),
                                   ("fp32 + CUDA Graph", False, True), ("fp16 + CUDA Graph", True, True)):
            r = {"ms": {}, "nmse_db": {}}
            G = load(tmpl.format(k=16), half)
            for N, reps in ((12, 300), (300, 50)):
                x = torch.randn(N, 2 * TP, 16, 16, device="cuda")
                try:
                    f = graphed(G, N) if graph else G
                    r["ms"][str(N)] = timeit(f, x, reps)
                except Exception as e:                      # CUDA Graph 캡처 불가 연산이 있으면 기록
                    r["ms"][str(N)] = None; r["error"] = repr(e)[:200]
            if r["ms"]["12"] is not None:
                for k in (1, 16):
                    Gk = load(tmpl.format(k=k), half)
                    num = den = 0.0; cache = {}
                    for d in Windows(te, TP, 16).eval_batches(16):
                        x, y = to_model_io(d, k)
                        if graph:
                            if x.shape[0] not in cache:
                                cache[x.shape[0]] = graphed(Gk, x.shape[0])
                            with torch.no_grad():
                                p = cache[x.shape[0]](x)
                        else:
                            with torch.no_grad():
                                p = Gk(x)
                        num += float(((p - y) ** 2).sum()); den += float((y ** 2).sum())
                    r["nmse_db"][str(k)] = 10 * np.log10(num / den)
            res[model][vname] = r
            fmt = lambda v: "실패" if v is None else f"{v:.2f} ms"
            print(f"{model:11s} {vname:18s} | 윈도우(배치12) {fmt(r['ms']['12']):>10} | 배치300 {fmt(r['ms']['300']):>10} | "
                  f"NMSE κ=1 k=1 {r['nmse_db'].get('1', float('nan')):.3f}, k=16 {r['nmse_db'].get('16', float('nan')):.3f}"
                  + (f" | {r['error']}" if "error" in r else ""), flush=True)
    json.dump(res, open("runs/bench_inference_p.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
