"""
make_tables_v9b.py -- v9b 결과표 (논문용 마크다운) -> results_v9b_tables.md
  v9b: TR 38.811 §6.9.1 관례에 따라 위성 기준 방위각 0° 고정(단일 기하), 고각 구간 split
  표 1  주 비교 (test): baseline 7종 vs BSC-UNet-I, 최고 baseline 대비 Δ (에피소드 bootstrap 95% CI)
  표 2  guard 확인
  표 3  LOS / NLOS (test)
  표 4  ablation: 고전 부분만 (ManifoldLinReg, ManifoldPoly2 = UNet 제거) vs BSC-UNet-I
  표 5  방위각 일반화 (한계 + v10 빔 정렬 모델): runs/eval_bearing_v9b_bscI.json, v10 결과
입력: runs/eval_v9b_mp2_{short,long}_{test,guard}.json (baseline 8종 + bscI)
"""
import json
import os

KS = [1, 2, 4, 8, 12, 16, 24, 32]
BASE7 = ["Persistence", "LMMSE", "Kalman", "LinReg_eig", "Poly2_blind", "BeamLinReg", "ManifoldLinReg"]
SHORT = {"Poly2_blind": "Poly2", "ManifoldLinReg": "ManifoldLinReg", "LinReg_eig": "LinReg"}
M = "bscI"


def load(prefix, split):
    G, P = {}, {}
    for part in ("short", "long"):
        f = f"runs/{prefix}_{part}_{split}.json"
        if os.path.exists(f):
            r = json.load(open(f, encoding="utf-8"))
            G.update(r["groups"]); P.update(r["paired"])
    return G, P


def sig(ci):
    return "*" if ci[1] < 0 or ci[0] > 0 else ""


def fmt_d(p):
    return f"{p['delta_db']:+.2f}{sig(p['ci95'])}"


def rows_of(G, P, grp="all"):
    out = []
    for k in KS:
        g = G[str(k)][grp]
        best = min(BASE7, key=lambda m: g[m])
        out.append((k, g, best, P[f"{M}_k{k}"][grp][best]))
    return out


def write():
    L = []
    w = L.append
    G, P = load("eval_v9b_mp2", "test")
    rows = rows_of(G, P)
    w("# v9b 결과표 (BSC-UNet-I)\n")
    w("데이터: v9b (TR 38.811 NTN-CDL Urban, 고각 구간 split, 위성 기준 방위각 0° 고정 — TR 38.811 §6.9.1 관례), Tp = 20 ms, "
      "test 68 / guard 86 에피소드. NMSE [dB] (낮을수록 좋음). Δ = BSC-UNet-I − 비교 대상, "
      "* = 에피소드 bootstrap(10,000회) 95% 신뢰구간이 0을 포함하지 않음.\n")
    w("## 표 1. 주 비교 (test)\n")
    w("| 방법 | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|" + "---|" * len(KS))
    for m in BASE7:
        w(f"| {m} | " + " | ".join(f"{r[1][m]:.2f}" for r in rows) + " |")
    w("| **BSC-UNet-I (제안)** | " + " | ".join(f"**{r[1][f'{M}_k{r[0]}']:.2f}**" for r in rows) + " |")
    w("| 최고 baseline | " + " | ".join(SHORT.get(r[2], r[2]) for r in rows) + " |")
    w("| **Δ (vs 최고 baseline)** | " + " | ".join(f"**{fmt_d(r[3])}**" for r in rows) + " |")
    w("| 95% CI | " + " | ".join(f"[{r[3]['ci95'][0]:+.2f}, {r[3]['ci95'][1]:+.2f}]" for r in rows) + " |")
    w("| 제안 우세 에피소드 | " + " | ".join(f"{r[3]['rcg_wins']}/{r[3]['n_ep']}" for r in rows) + " |\n")
    Gg, Pg = load("eval_v9b_mp2", "guard")
    rg = rows_of(Gg, Pg)
    w("## 표 2. guard (train–test 완충 구간) 확인\n")
    w("| | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|" + "---|" * len(KS))
    w("| 최고 baseline | " + " | ".join(f"{r[1][r[2]]:.2f} ({SHORT.get(r[2], r[2])})" for r in rg) + " |")
    w("| BSC-UNet-I | " + " | ".join(f"{r[1][f'{M}_k{r[0]}']:.2f}" for r in rg) + " |")
    w("| **Δ** | " + " | ".join(f"**{fmt_d(r[3])}**" for r in rg) + " |")
    w("| 제안 우세 에피소드 | " + " | ".join(f"{r[3]['rcg_wins']}/{r[3]['n_ep']}" for r in rg) + " |\n")
    w("## 표 3. LOS / NLOS (test)\n")
    w("| 그룹 | | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|---|" + "---|" * len(KS))
    for grp in ("LOS", "NLOS"):
        rr = rows_of(G, P, grp)
        w(f"| {grp} | 최고 baseline | " + " | ".join(f"{r[1][r[2]]:.2f} ({SHORT.get(r[2], r[2])})" for r in rr) + " |")
        w(f"| {grp} | BSC-UNet-I | " + " | ".join(f"{r[1][f'{M}_k{r[0]}']:.2f}" for r in rr) + " |")
        w(f"| {grp} | Δ | " + " | ".join(fmt_d(r[3]) for r in rr) + " |")
    w("")
    w("## 표 4. ablation\n")
    w("ManifoldPoly2 = BSC-UNet에서 UNet을 제거한 고전 부분 (배열 매니폴드 빔 추정 + 빔계수 위상 2차/진폭 1차 외삽, 학습 없음). "
      "ManifoldLinReg = 같은 빔 + train으로 적합한 선형 필터.\n")
    for split, (Gs, Ps) in (("test", (G, P)), ("guard", (Gg, Pg))):
        w(f"### {split}\n")
        w("| 구성 | " + " | ".join(f"k={k}" for k in KS) + " |")
        w("|---|" + "---|" * len(KS))
        w("| 고전: 매니폴드 빔 + 선형 필터 (ManifoldLinReg) | " + " | ".join(f"{Gs[str(k)]['all']['ManifoldLinReg']:.2f}" for k in KS) + " |")
        w("| 고전: 매니폴드 빔 + 2차 외삽 (ManifoldPoly2, UNet 제거) | " + " | ".join(f"{Gs[str(k)]['all']['ManifoldPoly2']:.2f}" for k in KS) + " |")
        w("| **BSC-UNet-I (제안)** | " + " | ".join(f"**{Gs[str(k)]['all'][f'{M}_k{k}']:.2f}**" for k in KS) + " |")
        w("| Δ BSC-UNet-I − ManifoldPoly2 | " + " | ".join(fmt_d(Ps[f'{M}_k{k}']['all']['ManifoldPoly2']) for k in KS) + " |")
        w("| 95% CI | " + " | ".join(
            f"[{Ps[f'{M}_k{k}']['all']['ManifoldPoly2']['ci95'][0]:+.2f}, {Ps[f'{M}_k{k}']['all']['ManifoldPoly2']['ci95'][1]:+.2f}]"
            for k in KS) + " |\n")
    fb = "runs/eval_bearing_v9b_bscI.json"
    if os.path.exists(fb):
        B = json.load(open(fb, encoding="utf-8"))
        betas = list(B[next(iter(B))]["BSC-UNet-I"].keys())
        w("## 표 5. 방위각 일반화 (한계)\n")
        w("v9b test 채널에 열 방향 위상 기울기 e^{jβn}를 곱해 다른 방위각의 단말을 모사 (모델·baseline은 방위각 0°로 학습한 그대로). "
          "전체 NMSE [dB].\n")
        w("| k | 방법 | " + " | ".join(f"β={b}" for b in betas) + " |")
        w("|---|---|" + "---|" * len(betas))
        for k, byk in B.items():
            for m, byb in byk.items():
                w(f"| {k} | {m} | " + " | ".join(f"{byb[b]['전체']:.2f}" for b in betas) + " |")
        w("\n방위각을 포함한 v10 데이터(구면 궤도 기하)로 학습한 빔 정렬 모델 BSC-UNet-AI의 결과는 results_v10_tables.md 참조.\n")
    open("results_v9b_tables.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    write()
