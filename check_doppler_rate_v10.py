"""check_doppler_rate_v10.py -- v10 기하 Doppler·Doppler 변화율(생성기의 수치 미분)을 닫힌 식과 대조
  원궤도·정지 단말, L = r_sat - r_ue, d = |L|:  d' = V.L/d,  d'' = (|V|^2 + L.A - d'^2)/d,  A = -w^2 r_sat
  fD = -(fc/c) d',  fDdot = -(fc/c) d''
  (1) 시험 생성 에피소드에 저장된 fD_geo, fDdot_geo, offNadirDeg, satAzDeg vs 닫힌 식
  (2) 생성기 passGeometry(중앙 차분 1 ms)를 그대로 옮긴 계산 vs 닫힌 식, 무작위 기하 20000개
"""
import glob
import sys

import h5py
import numpy as np

Re, alt, vs, fc, c = 6371e3, 550e3, 7.6e3, 20e9, 299792458.0
Rh = Re + alt
w = vs / Rh


def central(e):
    x, g = Re / Rh, np.radians(e)
    return np.arccos(np.clip(x * np.cos(g) ** 2 + np.sin(g) * np.sqrt(np.maximum(1 - x ** 2 * np.cos(g) ** 2, 0)), -1, 1))


def vecs(th, psi):
    sat = Rh * np.array([np.sin(th), 0, np.cos(th)])
    ue = Re * np.array([0, np.sin(psi), np.cos(psi)])
    V = vs * np.array([np.cos(th), 0, -np.sin(th)])
    return sat, ue, V


def closed(th, psi):
    sat, ue, V = vecs(th, psi)
    L = sat - ue; d = np.linalg.norm(L)
    d1 = V @ L / d
    d2 = (V @ V + L @ (-w ** 2 * sat) - d1 ** 2) / d
    xb = V / vs; zb = -sat / Rh; yb = np.cross(zb, xb); u = -L / d
    up = ue / Re
    elev = np.degrees(np.arcsin(np.clip(L @ up / d, -1, 1)))
    return -(fc / c) * d1, -(fc / c) * d2, np.degrees(np.arctan2(u @ yb, u @ xb)), np.degrees(np.arccos(np.clip(u @ zb, -1, 1))), elev


def numeric(th, psi, dt=1e-3):                       # 생성기 passGeometry와 같은 중앙 차분
    f = lambda t: closed(t, psi)[0]
    return (f(th + w * dt) - f(th - w * dt)) / (2 * dt)


def theta_of(e, psi, sgn):
    return sgn * np.arccos(np.clip(np.cos(central(e)) / np.cos(psi), -1, 1))


D = sys.argv[1] if len(sys.argv) > 1 else "./csi_dataset_FDD_DL_Kaband_Urban_SCS60kHz_dt1ms_v10_quick"
print("(1) 저장된 에피소드 값 vs 닫힌 식")
print(f"{'elev':>6s} {'fD_geo[kHz]':>12s} {'닫힌식':>9s} | {'fDdot_geo[Hz/s]':>15s} {'닫힌식':>10s} {'상대오차':>9s} | {'eta 오차':>8s} {'az 오차':>8s} {'고각 재계산':>10s}")
for p in sorted(glob.glob(D + "/*/*.mat")):
    with h5py.File(p) as h:
        g = {k: float(np.asarray(h[k][()]).ravel()[0]) for k in ("elevDeg", "crossTrackDeg", "passSgn", "fD_geo", "fDdot_geo", "offNadirDeg", "satAzDeg")}
    psi = np.radians(g["crossTrackDeg"]); th = theta_of(g["elevDeg"], psi, g["passSgn"])
    fD, fDd, az, eta, el = closed(th, psi)
    rel = abs(g["fDdot_geo"] - fDd) / max(abs(fDd), 1e-9)
    daz = (g["satAzDeg"] - az + 180) % 360 - 180 if abs(fDd) > 0 and g["elevDeg"] < 89.99 else 0.0
    print(f"{g['elevDeg']:6.1f} {g['fD_geo']/1e3:12.3f} {fD/1e3:9.3f} | {g['fDdot_geo']:15.2f} {fDd:10.2f} {rel:9.2e} | {g['offNadirDeg']-eta:8.1e} {daz:8.1e} {el:10.3f}")

print("\n(2) 생성기 수치 미분 vs 닫힌 식, 무작위 기하 (고각 10-90, psi ~ U(-gamma, gamma), 통과 방향 +-1)")
rng = np.random.default_rng(0)
rel, absd, elerr = [], [], []
for _ in range(20000):
    e = rng.uniform(10, 89.5); gam = central(e); psi = gam * rng.uniform(-1, 1); sgn = rng.choice([-1, 1])
    th = theta_of(e, psi, sgn)
    _, fDd, _, _, el = closed(th, psi); num = numeric(th, psi)
    absd.append(abs(num - fDd)); rel.append(abs(num - fDd) / abs(fDd)); elerr.append(abs(el - e))
rel, absd = np.array(rel), np.array(absd)
print(f"  |수치 - 닫힌식| 최대 {absd.max():.2e} Hz/s, 상대오차 최대 {rel.max():.2e}, 중앙값 {np.median(rel):.2e}")
print(f"  기하 재계산 고각 오차 최대 {max(elerr):.2e} deg")
# 38.811 기준점: psi=0 천정 통과면 fDdot 최대 크기
_, f90, *_ = closed(0.0, 0.0)
print(f"  천정(psi=0) 닫힌식 fDdot = {f90:.1f} Hz/s  (참고: 생성기 v9b 함수 raw fDdot = -6449.3 Hz/s)")
