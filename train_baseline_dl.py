"""
train_dl_baseline.py -- LSTM / GRU / LSTM+CNN / Transformer를 RCG와 동일한 파이프라인으로 학습·평가
======================================================================================
RCG(Model7.py/train_rcg_mat.py)가 쓰는 것과 정확히 같은 data.py(Split/Windows/to_model_io/
from_model_out)를 그대로 재사용한다. 즉 causal 정규화, SNR∈[5,20]dB 노이즈 주입, Doppler 잔차
회전 추정·제거(doppler='est')가 RCG와 100% 동일하게 적용되므로 NMSE 비교가 공정하다.

평가지표는 프로젝트에서 계속 써온 "pooled-ratio NMSE_dB"다:
    NMSE_dB = 10*log10( sum_i |pred_i - true_i|^2 / sum_i |true_i|^2 )
배치별 NMSE를 평균하는 게 아니라, 전체 평가셋에 대해 오차에너지 합과 신호에너지 합을 각각 누적한
뒤 마지막에 한 번 비율을 취한다(배치 크기가 들쭉날쭉해도 왜곡되지 않음, §6.15/6.16과 동일 정의).

사용 예:
  python train_dl_baseline.py --data_root /path/to/csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9 \
      --models lstm,lstmcnn,gru,transformer --horizons 1,2,4,8,12,16 --tp 12

출력: 콘솔에 epoch별 train/val NMSE 로그 + 최종 요약표, --out_dir에 results.json/results.csv와
(옵션) 각 (모델,horizon) 조합의 best checkpoint(.pt) 저장.

주의(필독):
  - data.py는 RCG 쪽 파이프라인에서 그대로 가져온 파일입니다. 실제 사용 중인 파일과 한 글자라도
    다르면(특히 causal 정규화/노이즈/Doppler 보정 부분) RCG와의 비교가 깨집니다. 반드시 실제
    episode 로더 파일로 교체하거나 diff로 확인한 뒤 쓰세요.
  - Tp는 .mat 메타데이터의 Tp(파일 생성 시 고정값)와 다르게 줄 수 있지만(윈도우는 Windows(Tp=...)
    인자로 다시 자르므로), RCG와 비교하려면 RCG 학습에 쓴 Tp와 반드시 일치시켜야 합니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import Split, Windows, to_model_io, from_model_out, phase_augment
from models_dl import build_model, count_params


# ----------------------------------------------------------------------------
# 평가: pooled-ratio NMSE_dB (배치 평균이 아니라 에너지 누적 후 한 번에 비율)
# ----------------------------------------------------------------------------
@torch.no_grad()
def pooled_nmse_db(model: nn.Module, win: Windows, k_slot: int, device: torch.device,
                    windows_per_batch: int, seed: int, snr_db: float | None = None) -> float:
    model.eval()
    se = torch.zeros((), dtype=torch.float64, device=device)   # sum |pred-true|^2
    en = torch.zeros((), dtype=torch.float64, device=device)   # sum |true|^2
    for d in win.eval_batches(windows_per_batch=windows_per_batch, seed=seed, snr_db=snr_db):
        inp, tgt = to_model_io(d, k_slot)
        inp, tgt = inp.to(device), tgt.to(device)
        pred = model(inp)
        se += ((pred.double() - tgt.double()) ** 2).sum()
        en += (tgt.double() ** 2).sum()
    return (10.0 * torch.log10(se / en.clamp_min(1e-12))).item()


# ----------------------------------------------------------------------------
# 한 (모델, horizon) 조합 학습
# ----------------------------------------------------------------------------
def train_one(args, model_name: str, k_ms: float, sp_train: Split, sp_val: Split, sp_test: Split,
              device: torch.device):
    Tp = args.tp
    win_train = Windows(sp_train, Tp=Tp, Tf=sp_train.Tf_file, stride=args.train_stride, doppler=args.doppler)
    win_val = Windows(sp_val, Tp=Tp, Tf=sp_val.Tf_file, stride=1, doppler=args.doppler)
    win_test = Windows(sp_test, Tp=Tp, Tf=sp_test.Tf_file, stride=1, doppler=args.doppler)

    k_slot = round(k_ms / sp_train.dt_ms)
    assert 1 <= k_slot <= win_train.Tf, (
        f"horizon {k_ms}ms -> slot {k_slot} 이 Tf={win_train.Tf} 범위를 벗어납니다 "
        f"(dt={sp_train.dt_ms}ms 기준). --horizons 값을 확인하세요."
    )

    torch.manual_seed(args.seed)
    model = build_model(model_name, Tp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                         patience=max(1, args.patience // 2))

    g = torch.Generator(device=device if device.type == "cuda" else "cpu"); g.manual_seed(args.seed)
    perm_g = torch.Generator(); perm_g.manual_seed(args.seed + 1)   # 셔플 순열은 CPU에서 생성 (data.Windows.iterate)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, nb = 0.0, 0
        for d in win_train.iterate(args.batch_windows, shuffle=True, gen=g, perm_gen=perm_g, drop_last=True):
            inp, tgt = to_model_io(d, k_slot)
            inp, tgt = inp.to(device), tgt.to(device)
            if args.phase_aug:
                inp, tgt = phase_augment(inp, tgt, gen=g)
            opt.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = nn.functional.mse_loss(pred, tgt)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot_loss += loss.item()
            nb += 1

        val_nmse = pooled_nmse_db(model, win_val, k_slot, device, args.eval_batch_windows, args.eval_seed)
        sched.step(val_nmse)
        improved = val_nmse < best_val - args.min_delta
        flag = "  * best" if improved else ""
        print(f"[{model_name:11s} k={k_ms:>4.0f}ms] epoch {epoch:3d}/{args.epochs}  "
              f"train_mse={tot_loss / max(nb, 1):.5f}  val_NMSE={val_nmse:8.3f} dB"
              f"  ({time.time() - t0:5.0f}s){flag}")
        if improved:
            best_val = val_nmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  -> early stop (patience={args.patience} epochs without improvement)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_nmse = pooled_nmse_db(model, win_test, k_slot, device, args.eval_batch_windows, args.eval_seed)
    return model, best_val, test_nmse


# ----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=str, required=True,
                   help="MATLAB 생성기 outDir (train/val/test 서브폴더를 포함하는 경로)")
    p.add_argument("--models", type=str, default="lstm,lstmcnn",
                   help="쉼표로 구분된 모델 목록: lstm,gru,lstmcnn,transformer 중 선택")
    p.add_argument("--horizons", type=str, default="1,2,4,8,12,16",
                   help="평가할 horizon(ms) 목록, 쉼표 구분 (evalHorizonsMs와 맞추는 것을 권장)")
    p.add_argument("--tp", type=int, default=12, help="history 길이 Tp (RCG와 반드시 동일하게)")
    p.add_argument("--sc_stride", type=int, default=25, help="부반송파 서브샘플링 간격 (Split 생성자와 동일 기본값)")
    p.add_argument("--doppler", type=str, default="est", choices=["est", "est_parity", "none"],
                   help="Doppler 잔차 보정 방식 (v9 데이터는 'est' 권장)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8, help="val NMSE가 min_delta 이상 개선 안 되는 epoch 허용 수")
    p.add_argument("--min_delta", type=float, default=0.01, help="개선으로 인정할 최소 NMSE_dB 변화")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--batch_windows", type=int, default=24,
                   help="학습 시 한 스텝에 뽑는 (에피소드,시작) 윈도우 수 (실제 모델 배치=이값*n_sc)")
    p.add_argument("--eval_batch_windows", type=int, default=16, help="val/test 평가 배치 크기")
    p.add_argument("--eval_seed", type=int, default=12345, help="고정 평가 시드 (RCG 평가와 동일하게 유지)")
    p.add_argument("--train_stride", type=int, default=1, help="학습 윈도우 stride (1=전부 사용, 키우면 속도↑다양성↓)")
    p.add_argument("--phase_aug", action="store_true", help="학습 시 공통 위상회전 증강 사용")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_cache", action="store_true", help="Split의 .pt 캐시 파일을 쓰지 않음(디스크 변경 후 강제 재로드)")
    p.add_argument("--elev_range", type=float, nargs=2, default=None,
                   help="모든 split을 이 고각 범위 에피소드로 제한 (생성 중인 데이터로 비교할 때 제안 기법과 같은 에피소드 사용)")
    p.add_argument("--out_dir", type=str, default="./dl_baseline_runs")
    p.add_argument("--save_ckpt", action="store_true", help="(모델,horizon)별 best state_dict 저장")
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    sp_train = Split(args.data_root, "train", sc_stride=args.sc_stride, device=args.device, cache=not args.no_cache, elev_range=args.elev_range)
    sp_val = Split(args.data_root, "val", sc_stride=args.sc_stride, device=args.device, cache=not args.no_cache, elev_range=args.elev_range)
    sp_test = Split(args.data_root, "test", sc_stride=args.sc_stride, device=args.device, cache=not args.no_cache, elev_range=args.elev_range)
    print("train:", sp_train)
    print("val  :", sp_val)
    print("test :", sp_test)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    horizons = [float(h.strip()) for h in args.horizons.split(",") if h.strip()]

    results = []
    for model_name in model_names:
        for k_ms in horizons:
            print(f"\n===== {model_name}  horizon={k_ms}ms  (Tp={args.tp}) =====")
            model, val_nmse, test_nmse = train_one(args, model_name, k_ms, sp_train, sp_val, sp_test, device)
            n_params = count_params(model)
            row = {"model": model_name, "horizon_ms": k_ms, "tp": args.tp,
                   "val_nmse_db": val_nmse, "test_nmse_db": test_nmse, "params": n_params}
            results.append(row)
            print(f"  >>> {model_name} k={k_ms}ms  val_NMSE={val_nmse:.3f}dB  "
                  f"test_NMSE={test_nmse:.3f}dB  params={n_params:,}")
            if args.save_ckpt:
                ckpt_path = out_dir / f"{model_name}_k{int(k_ms)}ms_tp{args.tp}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  checkpoint saved: {ckpt_path}")

    # ---------------- 요약 ----------------
    print("\n================= 요약 (test NMSE_dB, 낮을수록 좋음) =================")
    header = f"{'model':12s} | " + " | ".join(f"k={h:>4.0f}ms" for h in horizons)
    print(header)
    print("-" * len(header))
    for model_name in model_names:
        vals = []
        for h in horizons:
            r = next(r for r in results if r["model"] == model_name and r["horizon_ms"] == h)
            vals.append(f"{r['test_nmse_db']:7.3f}")
        print(f"{model_name:12s} | " + " | ".join(f"{v:>6s}" for v in vals))

    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    json.dump(results, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "horizon_ms", "tp", "val_nmse_db", "test_nmse_db", "params"])
        w.writeheader()
        w.writerows(results)
    print(f"\n결과 저장: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
