"""validate_v10.py -- v10(기하 일관) 데이터 점검: 타이밍 점프 없음, 배열 위상 기울기 = π·sin(구면 off-nadir) (nadir 지향, 구면 기하),
기울기 방향 = 180° - 위성 기준 방위각, split별 방위각·cross-track 분포."""
import glob

import h5py
import numpy as np
import torch

from data import Split, fix_timing_jumps

D = "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10"


def main():
    for s in ("train", "val", "test", "guard"):
        sp = Split(D, s, fix_timing=False)
        _, sh = fix_timing_jumps(sp.X, sp.sc_idx)
        paths = sorted(glob.glob(f"{D}/{s}/episode_*.mat"))
        az, xt, err_mag, err_dir = [], [], [], []
        for e, p in enumerate(paths):
            with h5py.File(p, "r") as f:
                g = {k: float(np.asarray(f[k][()]).ravel()[0]) for k in ("satAzDeg", "crossTrackDeg", "elevDeg", "offNadirDeg")}
            X = sp.X[e, 0].to(torch.complex128).cpu().numpy()
            _, U = np.linalg.eigh(X.T @ X.conj()); v = U[:, -1].reshape(16, 16)
            a = np.angle((v[1:, :] * v[:-1, :].conj()).sum()); b = np.angle((v[:, 1:] * v[:, :-1].conj()).sum())
            az.append(g["satAzDeg"]); xt.append(g["crossTrackDeg"])
            if not sp.nlos[e] and g["elevDeg"] < 85:
                err_mag.append(abs(np.hypot(a, b) - np.pi * np.sin(np.deg2rad(g["offNadirDeg"]))))
                d = np.degrees(np.arctan2(b, a)) - (180 - g["satAzDeg"])
                err_dir.append(abs((d + 180) % 360 - 180))
        az = np.array(az)
        print(f"{s:5s}: E={sp.E} 타이밍 점프 {int((sh != 0).sum())}건 | LOS 기울기 크기 오차 중앙값 {np.median(err_mag):.3f} rad, "
              f"방향 오차 중앙값 {np.median(err_dir):.1f}° | 위성 기준 방위각 범위 [{az.min():.0f}, {az.max():.0f}]°, "
              f"|cross-track| 중앙값 {np.median(np.abs(xt)):.2f}°", flush=True)


if __name__ == "__main__":
    main()
