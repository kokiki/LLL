"""
make_tables_final.py -- 최종 결과표 (제안 모델 BSC-UNet-P, v9b) -> results_final_tables.md
  표 1  주 비교 (test): baseline 7종 vs BSC-UNet-P, 최고 baseline 대비 Δ (에피소드 bootstrap 95% CI)
  표 2  guard 확인
  표 3  LOS / NLOS (test)
  표 4  ablation (test·guard): 선형 지름길만 / 2차 외삽 지름길만 (학습 없음) / 선형 지름길 + UNet (BSC-UNet-I) / 제안
  표 5  Doppler 변화율 스케일 κ 민감도 (k = 1, 16; test·guard)
  표 6  방위각 일반화 시험 (한계)
입력: runs/eval_v9b_final_short_{test,guard}.json (baseline 8종 + bscP + bscI, k = 1–16),
      runs/eval_v9b_bscP{,_kappa1}_{test,guard}.json (κ 민감도), runs/eval_bearing_v9b_final.json
"""
import json
import os

KS = [1, 2, 4, 8, 12, 16]                     # k = 24, 32는 학습하지 않음
BASE7 = ["Persistence", "LMMSE", "Kalman", "LinReg_eig", "Poly2_blind", "BeamLinReg", "ManifoldLinReg"]
SHORT = {"Poly2_blind": "Poly2", "ManifoldLinReg": "ManifoldLinReg", "LinReg_eig": "LinReg"}


def load(files):
    G, P = {}, {}
    for f in files:
        if os.path.exists(f):
            r = json.load(open(f, encoding="utf-8"))
            for k, v in r["groups"].items():
                G.setdefault(k, {}).update(v)
            for m, v in r["paired"].items():
                P.setdefault(m, {}).update(v)
    return G, P


def sig(ci):
    return "*" if ci[1] < 0 or ci[0] > 0 else ""


def fd(p):
    return f"{p['delta_db']:+.2f}{sig(p['ci95'])}"


def main_rows(G, P, grp="all"):
    out = []
    for k in KS:
        g = G[str(k)][grp]
        best = min(BASE7, key=lambda m: g[m])
        out.append((k, g, best, P[f"bscP_k{k}"][grp][best]))
    return out


def write():
    L = []
    w = L.append
    T = {s: load([f"runs/eval_v9b_final_short_{s}.json"]) for s in ("test", "guard")}
    G, P = T["test"]
    rows = main_rows(G, P)
    w("# 최종 결과표 (제안 모델 BSC-UNet-P)\n")
    w("데이터: v9b (TR 38.811 NTN-CDL Urban, 고각 구간 split, 위성 기준 방위각 0° 고정, Doppler 변화율 κ = 0.155), Tp = 20 ms, "
      "test 68 / guard 86 에피소드. NMSE [dB] (낮을수록 좋음). Δ = BSC-UNet-P − 비교 대상, "
      "* = 에피소드 bootstrap(10,000회) 95% 신뢰구간이 0을 포함하지 않음. 학습률: 코사인 80 epoch 후 1e-6 유지.\n")
    w("## 표 1. 주 비교 (test)\n")
    w("| 방법 | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|" + "---|" * len(KS))
    for m in BASE7:
        w(f"| {m} | " + " | ".join(f"{r[1][m]:.2f}" for r in rows) + " |")
    w("| **BSC-UNet-P (제안)** | " + " | ".join(f"**{r[1][f'bscP_k{r[0]}']:.2f}**" for r in rows) + " |")
    w("| 최고 baseline | " + " | ".join(SHORT.get(r[2], r[2]) for r in rows) + " |")
    w("| **Δ (vs 최고 baseline)** | " + " | ".join(f"**{fd(r[3])}**" for r in rows) + " |")
    w("| 95% CI | " + " | ".join(f"[{r[3]['ci95'][0]:+.2f}, {r[3]['ci95'][1]:+.2f}]" for r in rows) + " |")
    w("| 제안 우세 에피소드 | " + " | ".join(f"{r[3]['rcg_wins']}/{r[3]['n_ep']}" for r in rows) + " |\n")
    Gg, Pg = T["guard"]
    rg = main_rows(Gg, Pg)
    w("## 표 2. guard (train–test 완충 구간) 확인\n")
    w("| | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|" + "---|" * len(KS))
    w("| 최고 baseline | " + " | ".join(f"{r[1][r[2]]:.2f} ({SHORT.get(r[2], r[2])})" for r in rg) + " |")
    w("| BSC-UNet-P | " + " | ".join(f"{r[1][f'bscP_k{r[0]}']:.2f}" for r in rg) + " |")
    w("| **Δ** | " + " | ".join(f"**{fd(r[3])}**" for r in rg) + " |")
    w("| 제안 우세 에피소드 | " + " | ".join(f"{r[3]['rcg_wins']}/{r[3]['n_ep']}" for r in rg) + " |\n")
    w("## 표 3. LOS / NLOS (test)\n")
    w("| 그룹 | | " + " | ".join(f"k={k}" for k in KS) + " |")
    w("|---|---|" + "---|" * len(KS))
    for grp in ("LOS", "NLOS"):
        rr = main_rows(G, P, grp)
        w(f"| {grp} | 최고 baseline | " + " | ".join(f"{r[1][r[2]]:.2f} ({SHORT.get(r[2], r[2])})" for r in rr) + " |")
        w(f"| {grp} | BSC-UNet-P | " + " | ".join(f"{r[1][f'bscP_k{r[0]}']:.2f}" for r in rr) + " |")
        w(f"| {grp} | Δ | " + " | ".join(fd(r[3]) for r in rr) + " |")
    w("")
    w("## 표 4. ablation\n")
    w("빔 추정(배열 매니폴드)은 모두 같고 빔 계수 예측과 UNet 유무만 다르다. "
      "딥러닝만 = 빔 추정·지름길 없이 UNet이 history에서 직접 예측 (k = 1, 16만 학습), "
      "ManifoldLinReg = 선형 지름길만 (UNet 없음, 닫힌 식 적합), ManifoldPoly2 = 2차 외삽 지름길만 (학습 없음), "
      "BSC-UNet-I = 선형 지름길 + UNet, BSC-UNet-P = 선형 + 2차 외삽 지름길 + UNet (제안). "
      "P − I는 pooled NMSE 차이 (신뢰구간 없음).\n")
    for split in ("test", "guard"):
        Gs, Ps = T[split]
        Gu, Pu = load([f"runs/eval_v9b_unet_{split}.json"])          # 딥러닝만 (지름길 없는 UNet, k = 1, 16)
        w(f"### {split}\n")
        w("| 구성 | " + " | ".join(f"k={k}" for k in KS) + " |")
        w("|---|" + "---|" * len(KS))
        w("| 딥러닝만 (지름길 없는 UNet) | " + " | ".join(
            (f"{Gu[str(k)]['all'][f'unet_k{k}']:.2f}" if str(k) in Gu and f"unet_k{k}" in Gu[str(k)]["all"] else "–") for k in KS) + " |")
        for lab, key in (("선형 지름길만 (ManifoldLinReg)", "ManifoldLinReg"), ("2차 외삽 지름길만 (ManifoldPoly2)", "ManifoldPoly2")):
            w(f"| {lab} | " + " | ".join(f"{Gs[str(k)]['all'][key]:.2f}" for k in KS) + " |")
        w("| 선형 지름길 + UNet (BSC-UNet-I) | " + " | ".join(f"{Gs[str(k)]['all'][f'bscI_k{k}']:.2f}" for k in KS) + " |")
        w("| **선형 + 2차 외삽 지름길 + UNet (BSC-UNet-P)** | " + " | ".join(f"**{Gs[str(k)]['all'][f'bscP_k{k}']:.2f}**" for k in KS) + " |")
        w("| Δ P − ManifoldPoly2 | " + " | ".join(fd(Ps[f'bscP_k{k}']['all']['ManifoldPoly2']) for k in KS) + " |")
        w("| Δ P − BSC-UNet-I | " + " | ".join(f"{Gs[str(k)]['all'][f'bscP_k{k}'] - Gs[str(k)]['all'][f'bscI_k{k}']:+.2f}" for k in KS) + " |\n")
    w("## 표 5. Doppler 변화율 스케일 κ 민감도\n")
    w("κ = 0.155: v9b 기본. κ = 1: 보상 없는 실제 Doppler 변화율 (저장 채널에 빠진 2차 위상을 곱해 생성). "
      "각 조건에서 baseline은 같은 κ의 train/val로 적합하고, 모델은 같은 κ 데이터로 학습했다.\n")
    w("| κ | split | k | Poly2_blind | ManifoldLinReg | ManifoldPoly2 | BSC-UNet-I | **BSC-UNet-P** | Δ P − ManifoldPoly2 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for kap, tag in (("0.155", "eval_v9b_bscP"), ("1", "eval_v9b_bscP_kappa1")):
        for split in ("test", "guard"):
            Gk, Pk = load([f"runs/{tag}_{split}.json"])
            for k in (1, 16):
                g = Gk[str(k)]["all"]
                w(f"| {kap} | {split} | {k} | {g['Poly2_blind']:.2f} | {g['ManifoldLinReg']:.2f} | {g['ManifoldPoly2']:.2f} | "
                  f"{g[f'bscI_k{k}']:.2f} | **{g[f'bscP_k{k}']:.2f}** | {fd(Pk[f'bscP_k{k}']['all']['ManifoldPoly2'])} |")
    w("")
    fb = "runs/eval_bearing_v9b_final.json"
    if os.path.exists(fb):
        B = json.load(open(fb, encoding="utf-8"))
        betas = list(B[next(iter(B))]["BSC-UNet-P"].keys())
        w("## 표 6. 방위각 일반화 시험 (한계)\n")
        w("v9b test 채널에 열 방향 위상 기울기 e^{jβn}를 곱해 다른 방위각의 단말을 모사 (모델·baseline은 방위각 0°로 학습한 그대로). 전체 NMSE [dB].\n")
        w("| k | 방법 | " + " | ".join(f"β={b}" for b in betas) + " |")
        w("|---|---|" + "---|" * len(betas))
        for k, byk in B.items():
            for m, byb in byk.items():
                w(f"| {k} | {m} | " + " | ".join(f"{byb[b]['전체']:.2f}" for b in betas) + " |")
        w("")
    open("results_final_tables.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    write()
