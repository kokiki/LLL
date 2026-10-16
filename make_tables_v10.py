"""
make_tables_v10.py -- v10 최종 결과표 (논문용 마크다운) -> results_v10_tables.md
  표 1  주 비교 (test): baseline 7종 vs BSC-UNet-AI, horizon별 NMSE, 최고 baseline 대비 Δ (에피소드 bootstrap 95% CI)
  표 2  guard 확인
  표 3  LOS / NLOS (test)
  표 4  ablation (test·guard): 고전 부분만 (ManifoldLinReg, ManifoldPoly2 = UNet 제거) / 정렬 없음 / 방향 증강 / 빔 정렬
입력: runs/eval_v10_bscAI_{short,long}_{test,guard}.json (baseline 7종 + AI + I),
      runs/eval_v10_mp2_{short,long}_{test,guard}.json (ManifoldPoly2 포함, AI vs ManifoldPoly2 paired),
      runs/eval_v10_fix_{test,guard}.json (BSC-UNet-DA, k=1·16)
"""
import json
import os

KS = [1, 2, 4, 8, 12, 16, 24, 32]
BASE7 = ["Persistence", "LMMSE", "Kalman", "LinReg_eig", "Poly2_blind", "BeamLinReg", "ManifoldLinReg"]
SHORT = {"Poly2_blind": "Poly2", "ManifoldLinReg": "ManifoldLinReg", "LinReg_eig": "LinReg"}


def load(prefix, split):
    G, P = {}, {}
    for part in ("short", "long"):
        f = f"runs/{prefix}_{part}_{split}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f, encoding="utf-8"))
        G.update(r["groups"]); P.update(r["paired"])
    return G, P


def sig(ci):
    return "*" if ci[1] < 0 or ci[0] > 0 else ""


def fmt_d(p):
    return f"{p['delta_db']:+.2f}{sig(p['ci95'])}"


def main_tables(split):
    G, P = load("eval_v10_bscAI", split)
    rows = []
    for k in KS:
        g = G[str(k)]["all"]
        best = min(BASE7, key=lambda m: g[m])
        p = P[f"bscAI_k{k}"]["all"][best]
        rows.append((k, g, best, p))
    return G, P, rows


def write():
    L = []
    w = L.append
    G, P, rows = main_tables("test")
    w("# v10 결과표 (BSC-UNet-AI)\n")
    w("데이터: v10 (구면 궤도 기하, 위성 기준 방위각·Doppler·Doppler 변화율 일관, 16x16 UPA nadir 지향), Tp = 20 ms, "
      "test 68 에피소드 / guard 86 에피소드. NMSE [dB] (낮을수록 좋음). Δ = BSC-UNet-AI − 비교 대상, "
      "* = 에피소드 bootstrap(10,000회) 95% 신뢰구간이 0을 포함하지 않음.\n")
    # 표 1
    w("## 표 1. 주 비교 (test)\n")
    w("| 방법 | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|" + "---|" * len(KS))
    for m in BASE7:
        w(f"| {m} | " + " | ".join(f"{r[1][m]:.2f}" for r in rows) + " |")
    w("| **BSC-UNet-AI (제안)** | " + " | ".join(f"**{r[1][f'bscAI_k{r[0]}']:.2f}**" for r in rows) + " |")
    w("| 최고 baseline | " + " | ".join(SHORT.get(r[2], r[2]) for r in rows) + " |")
    w("| **Δ (vs 최고 baseline)** | " + " | ".join(f"**{fmt_d(r[3])}**" for r in rows) + " |")
    w("| 95% CI | " + " | ".join(f"[{r[3]['ci95'][0]:+.2f}, {r[3]['ci95'][1]:+.2f}]" for r in rows) + " |")
    w("| 제안 우세 에피소드 | " + " | ".join(f"{r[3]['rcg_wins']}/{r[3]['n_ep']}" for r in rows) + " |\n")
    # 표 2
    Gg, Pg, rg = main_tables("guard")
    w("## 표 2. guard (train–test 완충 구간) 확인\n")
    w("| | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|" + "---|" * len(KS))
    w("| 최고 baseline | " + " | ".join(f"{r[1][r[2]]:.2f} ({SHORT.get(r[2], r[2])})" for r in rg) + " |")
    w("| BSC-UNet-AI | " + " | ".join(f"{r[1][f'bscAI_k{r[0]}']:.2f}" for r in rg) + " |")
    w("| **Δ** | " + " | ".join(f"**{fmt_d(r[3])}**" for r in rg) + " |")
    w("| 제안 우세 에피소드 | " + " | ".join(f"{r[3]['rcg_wins']}/{r[3]['n_ep']}" for r in rg) + " |\n")
    # 표 3
    w("## 표 3. LOS / NLOS (test)\n")
    w("| 그룹 | | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|---|" + "---|" * len(KS))
    for grp in ("LOS", "NLOS"):
        bests = [min(BASE7, key=lambda m: G[str(k)][grp][m]) for k in KS]
        w(f"| {grp} | 최고 baseline | " + " | ".join(f"{G[str(k)][grp][b]:.2f} ({SHORT.get(b, b)})" for k, b in zip(KS, bests)) + " |")
        w(f"| {grp} | BSC-UNet-AI | " + " | ".join(f"{G[str(k)][grp][f'bscAI_k{k}']:.2f}" for k in KS) + " |")
        w(f"| {grp} | Δ | " + " | ".join(fmt_d(P[f'bscAI_k{k}'][grp][b]) for k, b in zip(KS, bests)) + " |")
    w("")
    # 표 4
    w("## 표 4. ablation\n")
    w("ManifoldPoly2 = BSC-UNet에서 UNet을 제거한 고전 부분(배열 매니폴드 빔 추정 + 빔계수 위상 2차/진폭 1차 외삽, 학습 없음). "
      "ManifoldLinReg = 같은 빔 + train으로 적합한 선형 필터. BSC-UNet-I = 빔 정렬 없음, BSC-UNet-DA = 정렬 없음 + 방향 증강 "
      "(k=1, 16만 학습).\n")
    for split in ("test", "guard"):
        Gm, Pm = load("eval_v10_mp2", split)
        Ga, _ = load("eval_v10_bscAI", split)
        Gd = {}
        f = f"runs/eval_v10_fix_{split}.json"
        if os.path.exists(f):
            Gd = json.load(open(f, encoding="utf-8"))["groups"]
        w(f"### {split}\n")
        w("| 구성 | " + " | ".join(f"k={k}" for k in KS) + " |")
        w("|---|" + "---|" * len(KS))
        w("| 고전: 매니폴드 빔 + 선형 필터 (ManifoldLinReg) | " + " | ".join(f"{Gm[str(k)]['all']['ManifoldLinReg']:.2f}" for k in KS) + " |")
        w("| 고전: 매니폴드 빔 + 2차 외삽 (ManifoldPoly2, UNet 제거) | " + " | ".join(f"{Gm[str(k)]['all']['ManifoldPoly2']:.2f}" for k in KS) + " |")
        w("| BSC-UNet-I (빔 정렬 없음) | " + " | ".join(f"{Ga[str(k)]['all'][f'bscI_k{k}']:.2f}" for k in KS) + " |")
        w("| BSC-UNet-DA (정렬 없음 + 방향 증강) | " + " | ".join(
            (f"{Gd[str(k)]['all'][f'bscDA_k{k}']:.2f}" if str(k) in Gd and f"bscDA_k{k}" in Gd[str(k)]["all"] else "–") for k in KS) + " |")
        w("| **BSC-UNet-AI (빔 정렬, 제안)** | " + " | ".join(f"**{Gm[str(k)]['all'][f'bscAI_k{k}']:.2f}**" for k in KS) + " |")
        w("| Δ AI − ManifoldPoly2 | " + " | ".join(fmt_d(Pm[f'bscAI_k{k}']['all']['ManifoldPoly2']) for k in KS) + " |")
        w("| 95% CI | " + " | ".join(
            f"[{Pm[f'bscAI_k{k}']['all']['ManifoldPoly2']['ci95'][0]:+.2f}, {Pm[f'bscAI_k{k}']['all']['ManifoldPoly2']['ci95'][1]:+.2f}]" for k in KS) + " |\n")
    open("results_v10_tables.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    write()
