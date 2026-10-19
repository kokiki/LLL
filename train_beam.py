"""
train_beam.py -- BSC-UNet 학습: 배열 구조 빔 지름길 + 비선형 UNet 보정 (train.py 복사본, 모델만 교체)
==========================================================================
모델  : csi_models.BeamShortcutUNet  G(x) = (빔 영역 선형 예측) + UNet([x, 빔 투영 x])
        --aligned: csi_models.BeamAlignedUNet (빔 정렬 좌표 UNet, v10의 BSC-UNet-AI)
        (아래 설명 중 모델 외 부분은 train.py와 동일)
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

from data import Split, Windows, phase_augment, rescale_doppler_rate, to_model_io


def array_resp(B, device, gen=None):
    """윈도우별 무작위 위성 배열 소자 응답 (B, 256) 복소: c = 10^(g/20) e^{jφ}, φ ~ N(0, σφ²), g ~ N(0, σg²),
    σφ ~ U[0, 20°], σg ~ U[0, 2 dB]; 30% 확률로 고장 소자(응답 0) 비율 ~ U[0, 10%]; 20%는 이상적 배열 (c = 1)."""
    u = lambda *sh: torch.rand(*sh, device=device, generator=gen)
    n = lambda *sh: torch.randn(*sh, device=device, generator=gen)
    sph, sg = math.radians(20) * u(B, 1), 2.0 * u(B, 1)
    c = torch.polar(10 ** (n(B, 256) * sg / 20), n(B, 256) * sph)
    fail = (u(B, 1) < 0.3) & (u(B, 256) < 0.1 * u(B, 1))
    c = torch.where(fail, torch.zeros_like(c), c)
    ideal = u(B, 1) < 0.2
    return torch.where(ideal, torch.ones_like(c), c).to(torch.complex64)


def dir_augment(inp, tgt, gen=None):
    """방향 증강: 위성 쪽 채널은 모든 경로가 같은 방향의 평면파(rank-1)라, 배열 위상 기울기 e^{j(αm+βn)}를 곱하면
    다른 방향(방위각·off-nadir)의 단말 채널이 된다. 샘플별 α, β ~ U[-π, π)를 입력 모든 프레임과 정답에 같이 곱함
    (시간 구조·잡음·정규화는 그대로, train 데이터만 사용)."""
    n, C, H, W = inp.shape
    ab = 2 * math.pi * torch.rand(n, 2, device=inp.device, generator=gen) - math.pi
    m = torch.arange(H, device=inp.device, dtype=inp.dtype)[:, None]
    q = torch.arange(W, device=inp.device, dtype=inp.dtype)[None, :]
    ph = ab[:, 0, None, None] * m + ab[:, 1, None, None] * q          # (n, H, W)
    c, s = torch.cos(ph)[:, None], torch.sin(ph)[:, None]

    def rot(z, F):
        z = z.view(n, F, 2, H, W)
        re, im = z[:, :, 0], z[:, :, 1]
        return torch.stack([re * c - im * s, re * s + im * c], 2).reshape(n, 2 * F, H, W)
    return rot(inp, C // 2), rot(tgt, tgt.shape[1] // 2)
from csi_models import BeamAlignedPolyUNet, BeamAlignedUNet, BeamShortcutPolyUNet, BeamShortcutUNet, GeneratorUNet


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
    ap.add_argument("--aligned", action="store_true", help="BSC-UNet-A: 빔 정렬 좌표 UNet + replicate 패딩 (가장자리 효과 완화)")
    ap.add_argument("--poly_shortcut", action="store_true",
                    help="BSC-UNet-P: 빔 지름길에 빔계수 위상 2차/진폭 1차 외삽(ManifoldPoly2와 같은 식)을 더함 (Doppler 변화율 2차 위상 대응)")
    ap.add_argument("--sym_out", action="store_true", help="BSC-UNet UNet 출력층을 대칭 구조로 (맨 윗행 가장자리 오차 완화)")
    ap.add_argument("--pad_replicate", action="store_true", help="BSC-UNet UNet 모든 합성곱 패딩을 replicate로 (가장자리 오차 완화)")
    ap.add_argument("--no_phase_aug", action="store_true")
    ap.add_argument("--array_aug", action="store_true",
                    help="배열 비이상성 증강: 윈도우별 무작위 소자 응답 (위상 rms ~U[0,20]deg, 이득 rms ~U[0,2]dB, 30%% 확률로 "
                         "고장 소자 ~U[0,10]%%, 20%%는 이상적)을 노이즈 주입 전 채널에 곱함 (train만)")
    ap.add_argument("--dir_aug", action="store_true",
                    help="방향 증강: 샘플별 무작위 2차원 위상 기울기 e^{j(αm+βn)} (α, β ~ U[-π, π))를 입력·정답에 같이 곱함 (train만)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--n_bottleneck_layers", type=int, default=3)
    ap.add_argument("--windows_per_batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr_sched", default="none", choices=["none", "cosine"], help="cosine: lr -> lr_min, cosine_epochs에 걸쳐 감쇠")
    ap.add_argument("--cosine_epochs", type=int, default=80)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--ema", type=float, default=0.0, help=">0이면 가중치 지수이동평균(EMA) decay. val 평가·best 저장·test에 EMA 가중치 사용")
    ap.add_argument("--grad_clip", type=float, default=0.0, help=">0이면 gradient norm clipping")
    ap.add_argument("--val_smooth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--elev_range", type=float, nargs=2, default=None, help="모든 split을 이 고각 범위로 제한 (smoke/비교용)")
    ap.add_argument("--doppler_rate_kappa", type=float, default=None,
                    help="v9b Doppler 변화율 잔차 스케일을 κ=0.155에서 이 값으로 바꿔 사용 (1 = 실제 변화율, 민감도 실험용)")
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
    if a.doppler_rate_kappa is not None:                                  # Doppler 변화율 스케일 민감도 (v9b 전용)
        for sp in (tr, va, te):
            rescale_doppler_rate(sp, a.doppler_rate_kappa)
    Tp = a.Tp or tr.Tp_file
    k = int(round(a.horizon_ms / tr.dt_ms))
    Tf = max(k, tr.Tf_file)
    W = {n: Windows(sp, Tp, Tf, a.win_stride, a.doppler) for n, sp in (("train", tr), ("val", va), ("test", te))}
    tag = f"beam_k{a.horizon_ms:g}ms_Tp{Tp}" + ("_noSC" if a.no_shortcut else "") + (f"_rcg{a.lambda_cycle:g}" if a.rcg_cycle else "")
    out = a.out_dir or f"./runs/{os.path.basename(os.path.normpath(a.data_root))}_{tag}"
    os.makedirs(out, exist_ok=True)
    print(f"train {tr}\nval {va}\ntest {te}\nTp={Tp} k={k} ({a.horizon_ms:g}ms) windows: "
          f"train={len(W['train'])} val={len(W['val'])} test={len(W['test'])}  load {time.time()-t0:.1f}s  out={out}",
          flush=True)

    Gcls = GeneratorUNet if a.no_shortcut else (BeamAlignedUNet if a.aligned else BeamShortcutUNet)
    gkw = {"sym_out": True} if a.sym_out else {}
    if a.pad_replicate:
        gkw["pad_mode"] = "replicate"
    if a.poly_shortcut:
        Gcls = BeamAlignedPolyUNet if a.aligned else BeamShortcutPolyUNet     # --aligned --poly_shortcut = BSC-UNet-AP
        G = Gcls(2 * Tp, k, 2, dropout=a.dropout, n_bottleneck_layers=a.n_bottleneck_layers).to(dev)
    else:
        G = Gcls(2 * Tp, 2, dropout=a.dropout, n_bottleneck_layers=a.n_bottleneck_layers, **gkw).to(dev)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(a.beta1, 0.999))
    # cosine: epoch e(0부터)에서 lr = lr_min + (lr - lr_min)(1 + cos(π e / cosine_epochs))/2, e >= cosine_epochs이면 lr_min 유지
    # (PyTorch CosineAnnealingLR은 T_max 이후 다시 증가하므로 사용하지 않음)
    rmin = a.lr_min / a.lr
    sched = (torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: rmin + (1 - rmin) * 0.5 * (1 + math.cos(math.pi * min(e, a.cosine_epochs) / a.cosine_epochs)))
             if a.lr_sched == "cosine" else None)
    if a.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        G_ema = AveragedModel(G, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    G_eval = (lambda: G_ema.module) if a.ema > 0 else (lambda: G)          # 평가·저장에 쓰는 가중치
    print(f"optim: Adam(lr={a.lr}, beta1={a.beta1}) sched={a.lr_sched} ema={a.ema} grad_clip={a.grad_clip}", flush=True)
    rcg = RCGLoss(G, Tp, dev)
    print(f"G={Gcls.__name__} params={sum(p.numel() for p in G.parameters()):,}  loss="
          f"{f'MSE + {a.lambda_cycle}*RCG cycle' if a.rcg_cycle else 'MSE'}  phase_aug={not a.no_phase_aug}  dir_aug={a.dir_aug}  array_aug={a.array_aug}",
          flush=True)

    g_noise = torch.Generator(device=dev); g_noise.manual_seed(a.seed)
    g_aug = torch.Generator(device=dev); g_aug.manual_seed(a.seed + 1)
    g_perm = torch.Generator(); g_perm.manual_seed(a.seed)
    logf = os.path.join(out, "log.csv")
    with open(logf, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nmse_db", "val_nmse_db", "val_smoothed_db"])
    best, best_raw, bad, hist, best_ep = float("inf"), None, 0, collections.deque(maxlen=a.val_smooth), 0
    for ep in range(1, a.epochs + 1):
        G.train(); t1 = time.time()
        tn = torch.zeros((), device=dev); td = torch.zeros((), device=dev)
        for d in W["train"].iterate(a.windows_per_batch, True, gen=g_noise, perm_gen=g_perm, drop_last=True,
                                    **({"elem_resp_fn": array_resp} if a.array_aug else {})):
            x, y = to_model_io(d, k)
            if not a.no_phase_aug:
                x, y = phase_augment(x, y, g_aug)
            if a.dir_aug:
                x, y = dir_augment(x, y, g_aug)
            if not a.rcg_cycle:
                p = G(x); loss = torch.mean((p - y) ** 2)
            else:
                l_rcg, p = rcg(x, y)
                mse = torch.mean((p - y) ** 2)
                loss = mse + a.lambda_cycle * (mse.detach() / (l_rcg.detach() + 1e-12)) * l_rcg
            opt.zero_grad(set_to_none=True); loss.backward()
            if a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), a.grad_clip)
            opt.step()
            if a.ema > 0:
                G_ema.update_parameters(G)
            tn += torch.sum((p.detach() - y) ** 2); td += torch.sum(y ** 2)
        tr_db = 10 * np.log10(tn.item() / td.item())
        if sched is not None:
            sched.step()
        v = nmse_db(G_eval(), W["val"], k); hist.append(v); sm = sum(hist) / len(hist)
        msg = f"Epoch {ep}/{a.epochs}  Train {tr_db:.2f}dB  Val {v:.2f}dB (smoothed {sm:.2f})  lr {opt.param_groups[0]['lr']:.1e}  {time.time()-t1:.0f}s"
        with open(logf, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr_db:.4f}", f"{v:.4f}", f"{sm:.4f}"])
        if sm < best:
            best, best_raw, bad, best_ep = sm, v, 0, ep
            torch.save(G_eval().state_dict(), os.path.join(out, "best_G.pt")); msg += "  *best"
        else:
            bad += 1
        print(msg, flush=True)
        if bad >= a.patience:
            print(f"early stopping at epoch {ep}", flush=True); break
    G.load_state_dict(torch.load(os.path.join(out, "best_G.pt"), weights_only=True))
    test_db = nmse_db(G, W["test"], k)                                       # (EMA 사용 시 best_G.pt = EMA 가중치)
    print(f"Test NMSE: {test_db:.2f} dB  (best smoothed val {best:.2f} dB)", flush=True)
    # vars(a)의 "epochs"(최대 epoch 인자)가 실제 값을 덮어쓰지 않도록 실제 학습 epoch·best epoch는 별도 키로 기록
    json.dump({"test_nmse_db": test_db, "best_val_smoothed_db": best, "best_val_raw_db": best_raw,
               "horizon_ms": a.horizon_ms, "k": k, "Tp": Tp, "dt_ms": tr.dt_ms, **vars(a),
               "epochs_trained": ep, "best_epoch": best_ep},
              open(os.path.join(out, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
