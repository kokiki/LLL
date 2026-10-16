"""
train.py -- 제안 기법 학습: 선형 시간 지름길 + 비선형 UNet 보정 (전부 GPU)
==========================================================================
모델  : csi_models.LinearShortcutUNet  G(x) = W*x + UNet(x)   (--no_shortcut: 순수 UNet, ablation)
손실  : MSE(x̂_{t+k}, x_{t+k})  (제안 기법, NMSE와 일치)
        --rcg_cycle (ablation): L = MSE + λ_eff·L_RCG,  λ_eff = λ·MSE.detach()/L_RCG.detach()
          L_RCG = RCG cycle-consistency (G 4회: 미래 예측, 뒤집은 시퀀스로 첫 프레임 복원, 각각의 재예측/재복원;
                  L1 + 0.005·LoG 유사도). v7 D안 k=14에서 MSE만과 test 동일(-19.13 vs -19.20dB) -> 제안 기법에서 제외
증강  : 공통 위상 회전 (train 배치만)
최적화: Adam lr 1e-4, β1 0.5, 배치 = 윈도우 4개 x subcarrier 12개, 3-epoch 평균 val로 best 저장, patience 15
데이터: data.Split/Windows (GPU 상주, GPU 배치 윈도우 생성). val은 고정 시드 배치.

Usage:
  python train.py --data_root ./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9 --horizon_ms 16
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch

from data import Split, Windows, phase_augment, to_model_io
from csi_models import GeneratorUNet, LinearShortcutUNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--horizon_ms", type=float, required=True)
    ap.add_argument("--Tp", type=int, default=None, help="history 관측 수 (기본: 파일 Tp)")
    ap.add_argument("--sc_stride", type=int, default=25)
    ap.add_argument("--doppler", default="est", choices=["est", "est_parity", "none"], help="est_parity: v7 이전 데이터 전용")
    ap.add_argument("--win_stride", type=int, default=1, help="에피소드 내 윈도우 시작 간격 [관측]")
    ap.add_argument("--rcg_cycle", action="store_true", help="ablation: MSE + λ·RCG cycle 손실 (기본은 MSE만)")
    ap.add_argument("--lambda_cycle", type=float, default=0.3)
    ap.add_argument("--no_shortcut", action="store_true", help="ablation: 선형 지름길 없는 순수 UNet")
    ap.add_argument("--no_phase_aug", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--n_bottleneck_layers", type=int, default=3)
    ap.add_argument("--windows_per_batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val_smooth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--elev_range", type=float, nargs=2, default=None, help="모든 split을 이 고각 범위로 제한 (smoke/비교용)")
    return ap.parse_args()


class RCGLoss:
    """train_rcg_mat.py의 RCG cycle 손실과 동일 정의 (frame = re/im 2채널)."""

    def __init__(self, G, Tp, device, w_LoG=0.005):
        self.G, self.Tp, self.w_LoG = G, Tp, w_LoG
        filtersize, sigma = 3, 1
        w = math.ceil(sigma * filtersize); wr = w // 2
        rr = torch.arange(-wr, wr + 1)
        Y, X = torch.meshgrid(rr, rr, indexing="xy")
        z = X.float() ** 2 + Y.float() ** 2
        fil = (z - 2 * sigma ** 2) * torch.exp(-z / 2 * (sigma ** 2)) / (2 * math.pi * sigma ** 6)
        self.fil = fil[None, None].to(device)

    def LoG(self, imgs):
        res = torch.nn.functional.conv2d(imgs, self.fil.repeat(1, imgs.size(1), 1, 1), padding="same").squeeze(1)
        mn, mx = res.min(), res.max()
        return ((res - mn) / (mx - mn)).unsqueeze(1) * 2 - 1

    @staticmethod
    def reverse(x):
        B, C, H, W = x.shape
        return x.view(B, C // 2, 2, H, W).flip(1).reshape(B, C, H, W)

    def __call__(self, x, y):
        G, Tp = self.G, self.Tp
        xm = x[:, :2]
        xm1_to_n1 = torch.cat((x[:, 2:], y), 1)
        y_d = G(x)
        xm_d = G(self.reverse(xm1_to_n1))
        y_dd = G(torch.cat((xm_d, x[:, 2:]), 1))
        xm_dd = G(self.reverse(torch.cat((xm1_to_n1[:, :2 * (Tp - 1)], y_d), 1)))
        l1 = lambda a, b: torch.mean(torch.abs(a - b))
        lg = lambda a, b: torch.mean(torch.abs(self.LoG(a) - self.LoG(b)))

        def terms(sim):
            return (sim(xm, xm_d) + sim(xm, xm_dd) + sim(xm_d, xm_dd) + sim(y_d, y_dd) + sim(y, y_d) + sim(y, y_dd))
        return terms(l1) + self.w_LoG * terms(lg), y_d


@torch.no_grad()
def nmse_db(G, windows, k, wpb=16, seed=12345):
    G.eval()
    num = den = 0.0
    for d in windows.eval_batches(wpb, seed=seed):
        x, y = to_model_io(d, k)
        num += torch.sum((G(x) - y) ** 2).item(); den += torch.sum(y ** 2).item()
    return 10 * np.log10(num / den)


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    tr, va, te = (Split(a.data_root, s, a.sc_stride, dev, elev_range=a.elev_range) for s in ("train", "val", "test"))
    Tp = a.Tp or tr.Tp_file
    k = int(round(a.horizon_ms / tr.dt_ms))
    Tf = max(k, tr.Tf_file)
    W = {n: Windows(sp, Tp, Tf, a.win_stride, a.doppler) for n, sp in (("train", tr), ("val", va), ("test", te))}
    tag = f"k{a.horizon_ms:g}ms_Tp{Tp}" + ("_noSC" if a.no_shortcut else "") + (f"_rcg{a.lambda_cycle:g}" if a.rcg_cycle else "")
    out = a.out_dir or f"./runs/{os.path.basename(os.path.normpath(a.data_root))}_{tag}"
    os.makedirs(out, exist_ok=True)
    print(f"train {tr}\nval {va}\ntest {te}\nTp={Tp} k={k} ({a.horizon_ms:g}ms) windows: "
          f"train={len(W['train'])} val={len(W['val'])} test={len(W['test'])}  load {time.time()-t0:.1f}s  out={out}",
          flush=True)

    Gcls = GeneratorUNet if a.no_shortcut else LinearShortcutUNet
    G = Gcls(2 * Tp, 2, dropout=a.dropout, n_bottleneck_layers=a.n_bottleneck_layers).to(dev)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, 0.999))
    rcg = RCGLoss(G, Tp, dev)
    print(f"G={Gcls.__name__} params={sum(p.numel() for p in G.parameters()):,}  loss="
          f"{f'MSE + {a.lambda_cycle}*RCG cycle' if a.rcg_cycle else 'MSE'}  phase_aug={not a.no_phase_aug}",
          flush=True)

    g_noise = torch.Generator(device=dev); g_noise.manual_seed(a.seed)
    g_aug = torch.Generator(device=dev); g_aug.manual_seed(a.seed + 1)
    g_perm = torch.Generator(); g_perm.manual_seed(a.seed)
    logf = os.path.join(out, "log.csv")
    with open(logf, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nmse_db", "val_nmse_db", "val_smoothed_db"])
    best, best_raw, bad, hist = float("inf"), None, 0, collections.deque(maxlen=a.val_smooth)
    for ep in range(1, a.epochs + 1):
        G.train(); t1 = time.time()
        tn = torch.zeros((), device=dev); td = torch.zeros((), device=dev)
        for d in W["train"].iterate(a.windows_per_batch, True, gen=g_noise, perm_gen=g_perm, drop_last=True):
            x, y = to_model_io(d, k)
            if not a.no_phase_aug:
                x, y = phase_augment(x, y, g_aug)
            if not a.rcg_cycle:
                p = G(x); loss = torch.mean((p - y) ** 2)
            else:
                l_rcg, p = rcg(x, y)
                mse = torch.mean((p - y) ** 2)
                loss = mse + a.lambda_cycle * (mse.detach() / (l_rcg.detach() + 1e-12)) * l_rcg
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tn += torch.sum((p.detach() - y) ** 2); td += torch.sum(y ** 2)
        tr_db = 10 * np.log10(tn.item() / td.item())
        v = nmse_db(G, W["val"], k); hist.append(v); sm = sum(hist) / len(hist)
        msg = f"Epoch {ep}/{a.epochs}  Train {tr_db:.2f}dB  Val {v:.2f}dB (smoothed {sm:.2f})  {time.time()-t1:.0f}s"
        with open(logf, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_db:.4f}", f"{v:.4f}", f"{sm:.4f}"])
        if sm < best:
            best, best_raw, bad = sm, v, 0
            torch.save(G.state_dict(), os.path.join(out, "best_G.pt")); msg += "  *best"
        else:
            bad += 1
        print(msg, flush=True)
        if bad >= a.patience:
            print(f"early stopping at epoch {ep}", flush=True); break
    G.load_state_dict(torch.load(os.path.join(out, "best_G.pt"), weights_only=True))
    test_db = nmse_db(G, W["test"], k)
    print(f"Test NMSE: {test_db:.2f} dB  (best smoothed val {best:.2f} dB)", flush=True)
    json.dump({"test_nmse_db": test_db, "best_val_smoothed_db": best, "best_val_raw_db": best_raw, "epochs": ep,
               "horizon_ms": a.horizon_ms, "k": k, "Tp": Tp, "dt_ms": tr.dt_ms, **vars(a)},
              open(os.path.join(out, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
