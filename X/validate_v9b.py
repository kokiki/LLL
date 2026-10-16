"""
validate_v9b.py -- v9b(타이밍 오프셋 에피소드 고정) 데이터 검증
  (1) v9b 원본에 타이밍 기준 점프가 없는지 (data.fix_timing_jumps 감지 수 = 0이어야 함)
  (2) 여러 subcarrier가 동시에 급변하는 스텝(모양 왜곡 포함)이 없는지
  (3) v9 + 로더 보정(A)과 v9b가 같은지: 같은 split·파일명 에피소드끼리 상대 오차 (A 보정이 정확했는지 검증)
"""
import sys

import numpy as np
import torch

from data import Split, fix_timing_jumps

V9 = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9"
V9B = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v9b"


def multi_sc_jumps(X):
    r = (X[:, :, 1:].conj() * X[:, :, :-1]).sum(-1)
    dphi = torch.angle(r)
    acc = torch.angle(torch.exp(1j * (dphi[:, :, 1:] - dphi[:, :, :-1])))
    return ((acc.abs() > 0.5).sum(1) >= 3)                                  # (E, T-2)


def main():
    tot_bad = 0
    for s in ("train", "val", "test", "guard"):
        b = Split(V9B, s, fix_timing=False)
        a = Split(V9, s)                                                       # v9 + A 보정
        _, sh = fix_timing_jumps(b.X, b.sc_idx)
        mj = multi_sc_jumps(b.X)
        E = min(a.E, b.E)                                                      # (생성 중간 점검 시 앞쪽 공통 에피소드만)
        same_meta = np.allclose(a.meta["elevDeg"][:E].astype(float), b.meta["elevDeg"][:E].astype(float)) and (a.nlos[:E] == b.nlos[:E]).all()
        rel = ((a.X[:E] - b.X[:E]).abs().pow(2).sum(dim=(1, 2, 3)) / b.X[:E].abs().pow(2).sum(dim=(1, 2, 3))).sqrt().cpu().numpy()
        diff = np.where(rel > 1e-2)[0]
        tot_bad += int((sh != 0).sum()) + int(mj.sum())
        print(f"{s:5s}: E={b.E}  v9b 점프 감지 {int((sh != 0).sum())}건, 다중 subcarrier 급변 {int(mj.sum())}건 | "
              f"메타(고각·LOS/NLOS) v9와 동일: {same_meta} | v9(A보정) vs v9b 상대오차 중앙값 {np.median(rel):.1e}, "
              f"최대 {rel.max():.1e}, 1% 초과 에피소드 {len(diff)}개", flush=True)
        for e in diff[:10]:
            print(f"      {b.names[e]} ({'NLOS' if b.nlos[e] else 'LOS'}, elev {b.meta['elevDeg'][e]:.1f}): 상대오차 {rel[e]:.3f}")
    print("검증 결과:", "통과 (v9b에 점프 없음)" if tot_bad == 0 else f"확인 필요 (이상 스텝 {tot_bad}건)")


if __name__ == "__main__":
    main()
