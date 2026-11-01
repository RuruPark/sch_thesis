# -*- coding: utf-8 -*-
# 추가 실험(17/18/19/20번) 결과로 논문용 그림을 만듦
# 입력: data/results/boot_*.csv, extra_fac_*_metrics.csv, recal_v2cmp.csv, extra_fac_v2cmp_folds.csv
# 출력: charts/thesis_fig*.png(300dpi), 같은 이름의 .pdf(벡터, 논문 본문용)
#
# 사용법: python scripts/21_thesis_figures.py [--dpi 300]
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT_ROOT, "data", "results")
OUT = os.path.join(PROJECT_ROOT, "charts")

# 02_eda.py와 같은 색. 규칙은 비교 기준이라 회색으로 뒤로 뺌
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID, RULE = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
WIDTH = 6.3  # 인치. A4 본문 폭(16cm)에 맞춤

PUMP_ONLY, PUMP_EXTRA = "LightGBM 펌프만", "LightGBM 펌프+추가"
# 모든 그림에서 모델 색을 고정해, 그림끼리 같은 모델을 같은 색으로 읽게 함
MODELS = [("규칙", RULE), ("Logistic Regression 펌프+추가", AQUA), (PUMP_ONLY, ORANGE),
          ("Random Forest 펌프+추가", VIOLET), (PUMP_EXTRA, BLUE)]
TYPES = {"3": "공기압축기", "4": "공작기계", "6": "가열로", "7": "압출기", "9": "주조기", "10": "열처리"}


def set_style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9,
                         "axes.titlesize": 10, "legend.frameon": False, "pdf.fonttype": 42})


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def save(fig, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("저장:", name)


def ci_row(df, model, group="전체"):
    r = df[(df["group"] == group) & (df["model"] == model)].iloc[0]
    return r["pr_auc"], r["ci_low"], r["ci_high"]


def diff_row(df, a, b, group="전체"):
    r = df[(df["group"] == group) & (df["model_a"] == a) & (df["model_b"] == b)].iloc[0]
    return r["diff"], r["ci_low"], r["ci_high"], r["p_b_better"]


def fig1_models(dpi):
    ci = read("boot_v2cmp_ci.csv").sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    y = np.arange(len(ci))
    colors = [dict(MODELS)[m] for m in ci["model"]]
    ax.hlines(y, ci["ci_low"], ci["ci_high"], color=colors, lw=2)
    ax.scatter(ci["pr_auc"], y, s=40, color=colors, zorder=3, edgecolor="white", linewidth=1.2)
    for yi, (v, hi) in enumerate(zip(ci["pr_auc"], ci["ci_high"])):
        ax.text(hi + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, ci["model"])
    ax.set_xlim(0, 0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC (처음 보는 펌프 설비 60대, 30분 앞)")
    ax.set_title("모델별 PR-AUC와 설비 단위 부트스트랩 95% 구간", loc="left")
    save(fig, "thesis_fig1_model_compare", dpi)


def fig2_horizon(dpi):
    tags = [("10분", "boot_v2cmp_h10_ci.csv"), ("30분", "boot_v2cmp_ci.csv"), ("60분", "boot_v2cmp_h60_ci.csv")]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    for k, (m, c) in enumerate(MODELS):
        xs = np.arange(len(tags)) + (k - 2) * 0.15
        vals = [ci_row(read(f), m) for _, f in tags]
        v, lo, hi = map(np.array, zip(*vals))
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", ms=6, color=c, ecolor=c, elinewidth=2,
                    capsize=0, mec="white", mew=1.2, label=m)
        if m == PUMP_EXTRA:
            for x, vi in zip(xs, v):
                ax.text(x + 0.06, vi, f"{vi:.3f}", va="center", ha="left", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(tags)), [f"{t} 앞 예측" for t, _ in tags])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("PR-AUC")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.set_title("예측 시간별 PR-AUC (95% 구간)", loc="left")
    save(fig, "thesis_fig2_horizon", dpi)


def fig3_gain_forest(dpi):
    rows = [("예측 10분 · 사건 5분", "boot_v2cmp_h10_diff.csv"), ("예측 30분 · 사건 5분 (기준)", "boot_v2cmp_diff.csv"),
            ("예측 60분 · 사건 5분", "boot_v2cmp_h60_diff.csv"), ("예측 30분 · 사건 3분", "boot_v2_d3_diff.csv"),
            ("예측 30분 · 사건 10분", "boot_v2_d10_diff.csv")]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for i, (label, f) in enumerate(rows[::-1]):
        d, lo, hi, p = diff_row(read(f), PUMP_ONLY, PUMP_EXTRA)
        c = BLUE if lo > 0 else MUTED
        ax.hlines(i, lo, hi, color=c, lw=2)
        ax.scatter(d, i, s=40, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(0.33, i, f"+{d:.3f}  [{lo:.3f}, {hi:.3f}]  p={p:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlim(-0.1, 0.32)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC 증가 (펌프만 → 펌프+추가)")
    ax.set_title("추가 설비종류 학습 효과와 95% 구간 (파랑: 구간이 0을 포함하지 않음)", loc="left")
    save(fig, "thesis_fig3_gain_forest", dpi)


def fig4_leave_one_out(dpi):
    m0 = read("extra_fac_v2cmp_metrics.csv").set_index("model")["pr_auc"]
    base, only = m0[PUMP_EXTRA], m0[PUMP_ONLY]
    rows = []
    for k, name in TYPES.items():
        m = read(f"extra_fac_v2_lo_{k}_metrics.csv").set_index("model")["pr_auc"][PUMP_EXTRA]
        rows.append((f"{k}.{name}", m))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    y = np.arange(len(rows))
    vals = np.array([r[1] for r in rows])
    ax.barh(y, vals, height=0.6, color=BLUE)
    for yi, v in zip(y, vals):
        # 기준선과 겹치지 않게 막대 안쪽 끝에 씀
        ax.text(v - 0.006, yi, f"{v:.3f} ({v - base:+.3f})", va="center", ha="right", fontsize=8, color="white")
    ax.axvline(base, color=INK, lw=1, ls="--")
    ax.axvline(only, color=ORANGE, lw=1, ls="--")
    ax.text(base, len(rows) - 0.4, f" 6종 전체 {base:.3f}", fontsize=8, color=INK, va="bottom")
    ax.text(only, len(rows) - 0.4, f" 펌프만 {only:.3f}", fontsize=8, color=ORANGE, va="bottom")
    ax.set_yticks(y, [f"{r[0]} 제외" for r in rows])
    ax.set_xlim(0, 0.75)
    ax.set_ylim(-0.6, len(rows) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC (LightGBM 펌프+추가, 30분 앞)")
    ax.set_title("설비종류를 하나씩 빼고 학습한 결과", loc="left")
    save(fig, "thesis_fig4_leave_one_out", dpi)


def fig5_by_facility(dpi):
    ci, diff = read("boot_v2s_ci.csv"), read("boot_v2s_diff.csv")
    same, pooled = "LightGBM 같은 종류만", "LightGBM 전체 종류"
    groups = sorted(ci["group"].unique(), key=lambda g: int(g.split(".")[0]))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.4), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(groups))
    for m, c, off in [("규칙", RULE, 0.22), (same, ORANGE, 0), (pooled, BLUE, -0.22)]:
        v = np.array([ci_row(ci, m, g) for g in groups])
        a1.errorbar(v[:, 0], y + off, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=5, color=c,
                    ecolor=c, elinewidth=1.6, capsize=0, mec="white", mew=1, label=m.replace("LightGBM ", ""))
    units = ci.drop_duplicates("group").set_index("group")["units"]
    a1.set_yticks(y, [f"{g} ({units[g]}대)" for g in groups])
    a1.set_xlim(0, 1)
    a1.set_xlabel("PR-AUC (95% 구간)")
    a1.grid(axis="y", visible=False)
    a1.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    for i, g in enumerate(groups):
        d, lo, hi, p = diff_row(diff, same, pooled, g)
        c = BLUE if lo > 0 else MUTED
        a2.hlines(i, lo, hi, color=c, lw=2)
        a2.scatter(d, i, s=36, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        a2.text(0.28, i, f"p={p:.3f}", va="center", fontsize=8, color=INK)
    a2.axvline(0, color=MUTED, lw=1)
    a2.set_xlim(-0.1, 0.33)
    a2.set_xlabel("PR-AUC 증가 (같은 종류만 → 전체 종류)")
    a2.grid(axis="y", visible=False)
    fig.suptitle("설비종류별 평가: 다른 종류를 함께 학습한 효과", x=0.01, ha="left", fontsize=10, y=1.0)
    save(fig, "thesis_fig5_by_facility", dpi)


def fig6_recal(dpi):
    r = read("recal_v2cmp.csv")
    r = r[(r["model"] == PUMP_EXTRA) & (r["days"] == 1)]
    short = {"기존": "기존", "점수 95% 분위 상한": "95%", "점수 98% 분위 상한": "98%",
             "점수 99% 분위 상한": "99%", "라벨 사용 F1": "라벨 F1"}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    # 2분 대기 점은 왼쪽에 이름을 달아, 낮은 오경보 쪽에 몰린 점끼리 글자가 겹치지 않게 함
    for w, c, dx, ha in [(1, BLUE, 5, "left"), (2, ORANGE, -5, "right")]:
        d = r[r["wait_min"] == w].set_index("method").loc[list(short)]
        ax.plot(d["false_alarm_per_day"], d["event_recall"], "o", ms=6, color=c, mec="white", mew=1.2,
                label=f"연속 {w}분 이상일 때 경보")
        for m, row in d.iterrows():
            ax.annotate(short[m], (row["false_alarm_per_day"], row["event_recall"]), xytext=(dx, 3),
                        textcoords="offset points", fontsize=8, color=INK, ha=ha)
    ax.set_xlim(-0.6, 8.5)
    ax.set_xlabel("하루 오경보 수")
    ax.set_ylabel("사건 재현율 (event recall)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"새 설비 첫 1일로 임계값 재보정 (LightGBM 펌프+추가, 평가 설비 {int(r['eval_units'].iloc[0])}대)",
                 loc="left")
    save(fig, "thesis_fig6_recalibration", dpi)


def fig8_alarm_wait(dpi):
    a = read("extra_fac_v2cmp_alarm.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    for m, c in MODELS:
        d = a[a["model"] == m].sort_values("wait_min")
        ax.plot(d["false_alarm_per_day"], d["event_recall"], "-o", color=c, lw=1.8, ms=5, mec="white", mew=1,
                label=m)
        if m in ("규칙", PUMP_EXTRA):
            for _, row in d.iterrows():
                ax.annotate(f"{int(row['wait_min'])}분", (row["false_alarm_per_day"], row["event_recall"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK)
    # 규칙은 1분 대기에서 하루 19회라 같은 축에 두면 나머지가 눌림 → 로그 축
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("하루 오경보 수 (로그 축)")
    ax.set_ylabel("사건 재현율 (event recall)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("경보 대기 시간(1·2·3·5·10분)별 사건 재현율과 오경보 (30분 앞)", loc="left")
    save(fig, "thesis_fig8_alarm_wait", dpi)


def fig7_folds(dpi):
    f = read("extra_fac_v2cmp_folds.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    for m, c in [("규칙", RULE), (PUMP_ONLY, ORANGE), (PUMP_EXTRA, BLUE)]:
        d = f[f["model"] == m].sort_values("fold")
        ax.plot(d["fold"], d["pr_auc"], "-o", color=c, lw=2, ms=6, mec="white", mew=1.2, label=m)
    ax.set_xticks(range(5), [f"묶음 {i}" for i in range(5)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ax.set_title("평가 묶음(설비 5묶음)별 PR-AUC", loc="left")
    save(fig, "thesis_fig7_folds", dpi)


def main():
    p = argparse.ArgumentParser(description="논문용 그림")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_style()
    for f in (fig1_models, fig2_horizon, fig3_gain_forest, fig4_leave_one_out, fig5_by_facility, fig6_recal,
              fig7_folds, fig8_alarm_wait):
        f(args.dpi)


if __name__ == "__main__":
    main()
