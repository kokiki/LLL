"""
train_baseline_dl.py -- 딥러닝 baseline (LSTM / GRU / LSTM+CNN / Transformer) 학습·평가
======================================================================================
제안 모델(BSC-UNet-P, train_beam.py)과 같은 데이터 조건·같은 학습 설정으로 학습해 공정하게 비교한다.
  데이터  : data.Split / Windows / to_model_io (인과적 정규화, SNR ~ U[5, 20] dB 잡음, Doppler 잔차 회전 제거),
            v9b, T_p = 20, Doppler 변화율 κ = 1 (주 결과; --doppler_rate_kappa, data.rescale_doppler_rate)
  학습    : Adam(lr 1e-4, β = (0.9, 0.999)), 코사인 80 epoch 후 lr_min 유지, EMA 0.999 (검증·저장·평가에 EMA 가중치),
            gradient clipping 1.0, 배치 = 윈도우 4개 x 부반송파 12개, 공통 위상 회전 증강,
            3-epoch 이동평균 val NMSE 최소 저장, patience 20, 최대 100 epoch, seed 42
  평가    : pooled NMSE_dB = 10 log10( Σ|pred − true|² / Σ|true|² ), evaluate.py와 같은 고정 시드 배치 (seed 12345, 16 윈도우)
  저장    : (모델, horizon)마다 --out_root/v9b_Tp{Tp}_{model}[_kappa1]_k{K}ms/ 에 best_G.pt, log.csv, result.json
            evaluate.py에서 --ckpt NAME=dl:<model>:<path>:<K> 로 baseline 9종·제안 모델과 함께 유의성 검정 가능

사용 예:
  python train_baseline_dl.py --models lstm,gru,lstmcnn,transformer --horizons 1,2,4,8,12,16
"""
from __future__ import annotations

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
import torch.nn as nn

from data import Split, Windows, phase_augment, rescale_doppler_rate, to_model_io
from models_dl import build_model, count_params


@torch.no_grad()
def pooled_nmse_db(model: nn.Module, win: Windows, k: int, wpb: int = 16, seed: int = 12345) -> float:
    """train_beam.nmse_db / evaluate.py와 같은 정의·같은 고정 시드 배치."""
    model.eval()
    num = den = 0.0
    for d in win.eval_batches(wpb, seed=seed):
        x, y = to_model_io(d, k)
        num += torch.sum((model(x) - y) ** 2).item(); den += torch.sum(y ** 2).item()
    return 10 * np.log10(num / den)


def train_one(a, name: str, k_ms: float, sps, dev):
    tr, va, te = sps
    Tp = a.tp
    k = int(round(k_ms / tr.dt_ms))
    Tf = max(k, tr.Tf_file)                                        # k > 16도 학습 가능 (train_beam.py와 동일)
    W = {n: Windows(sp, Tp, Tf, 1, a.doppler) for n, sp in (("train", tr), ("val", va), ("test", te))}
    tag = f"v9b_Tp{Tp}_{name}" + ("_kappa1" if a.doppler_rate_kappa == 1 else "") + f"_k{int(k_ms)}ms"
    out = os.path.join(a.out_root, tag); os.makedirs(out, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    G = build_model(name, Tp).to(dev)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, 0.999))
    rmin = a.lr_min / a.lr                                         # 코사인 cosine_epochs 후 lr_min 유지 (재상승 없음)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: rmin + (1 - rmin) * 0.5 * (1 + math.cos(math.pi * min(e, a.cosine_epochs) / a.cosine_epochs)))
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    G_ema = AveragedModel(G, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    g_noise = torch.Generator(device=dev); g_noise.manual_seed(a.seed)
    g_aug = torch.Generator(device=dev); g_aug.manual_seed(a.seed + 1)
    g_perm = torch.Generator(); g_perm.manual_seed(a.seed)
    print(f"\n===== {name} k={k_ms:g}ms (Tp={Tp}, κ={a.doppler_rate_kappa}) params={count_params(G):,} out={out} =====", flush=True)
    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nmse_db", "val_nmse_db", "val_smoothed_db"])
    best, best_raw, best_ep, bad, hist = float("inf"), None, 0, 0, collections.deque(maxlen=a.val_smooth)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        G.train(); tn = torch.zeros((), device=dev); td = torch.zeros((), device=dev)
        for d in W["train"].iterate(a.windows_per_batch, True, gen=g_noise, perm_gen=g_perm, drop_last=True):
            x, y = to_model_io(d, k)
            if not a.no_phase_aug:
                x, y = phase_augment(x, y, g_aug)
            p = G(x); loss = torch.mean((p - y) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward()
            if a.grad_clip > 0:
                nn.utils.clip_grad_norm_(G.parameters(), a.grad_clip)
            opt.step(); G_ema.update_parameters(G)
            tn += torch.sum((p.detach() - y) ** 2); td += torch.sum(y ** 2)
        sched.step()
        v = pooled_nmse_db(G_ema.module, W["val"], k); hist.append(v); sm = sum(hist) / len(hist)
        tr_db = 10 * np.log10(tn.item() / td.item())
        with open(os.path.join(out, "log.csv"), "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_db:.4f}", f"{v:.4f}", f"{sm:.4f}"])
        msg = f"[{name} k={k_ms:g}ms] epoch {ep:3d}  train {tr_db:7.2f} dB  val {v:7.2f} dB (smoothed {sm:7.2f})  lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:5.0f}s"
        if sm < best:
            best, best_raw, best_ep, bad = sm, v, ep, 0
            torch.save(G_ema.module.state_dict(), os.path.join(out, "best_G.pt")); msg += "  *best"
        else:
            bad += 1
        print(msg, flush=True)
        if bad >= a.patience:
            print(f"  early stopping at epoch {ep}", flush=True); break
    G.load_state_dict(torch.load(os.path.join(out, "best_G.pt"), weights_only=True))
    test_db = pooled_nmse_db(G, W["test"], k)
    res = {"model": name, "horizon_ms": k_ms, "k": k, "Tp": Tp, "test_nmse_db": test_db, "best_val_smoothed_db": best,
           "best_val_raw_db": best_raw, "params": count_params(G), **vars(a), "epochs_trained": ep, "best_epoch": best_ep}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=2)
    print(f"  >>> {name} k={k_ms:g}ms  test NMSE {test_db:.2f} dB  (best smoothed val {best:.2f} dB, epoch {best_ep})", flush=True)
    return res


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b")
    p.add_argument("--models", default="lstm,gru,lstmcnn,transformer", help="쉼표 구분: lstm, gru, lstmcnn, transformer")
    p.add_argument("--horizons", default="1,2,4,8,12,16", help="horizon(ms) 목록, 쉼표 구분")
    p.add_argument("--tp", type=int, default=20, help="history 길이 (제안 모델과 동일하게 20)")
    p.add_argument("--sc_stride", type=int, default=25)
    p.add_argument("--doppler", default="est", choices=["est", "est_parity", "none"])
    p.add_argument("--doppler_rate_kappa", type=float, default=1.0,
                   help="Doppler 변화율 잔차 스케일 (1 = 실제 변화율, 주 결과; 0.155 = 저장된 v9b 그대로)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--cosine_epochs", type=int, default=80)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--windows_per_batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_smooth", type=int, default=3)
    p.add_argument("--no_phase_aug", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", default="./runs")
    return p.parse_args()


def main():
    a = parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    sps = [Split(a.data_root, s, a.sc_stride, dev) for s in ("train", "val", "test")]
    if abs(a.doppler_rate_kappa - 0.1550556013060905) > 1e-6:
        for sp in sps:
            rescale_doppler_rate(sp, a.doppler_rate_kappa)
    print("\n".join(f"{n}: {sp}" for n, sp in zip(("train", "val", "test"), sps)), flush=True)
    names = [m.strip() for m in a.models.split(",") if m.strip()]
    horizons = [float(h) for h in a.horizons.split(",") if h.strip()]
    results = [train_one(a, n, h, sps, dev) for n in names for h in horizons]
    print("\n===== 요약: test NMSE [dB] =====")
    print(f"{'model':12s}" + "".join(f"  k={h:<5g}" for h in horizons))
    for n in names:
        print(f"{n:12s}" + "".join(f"{next(r['test_nmse_db'] for r in results if r['model'] == n and r['horizon_ms'] == h):9.2f}"
                                   for h in horizons))


if __name__ == "__main__":
    main()
