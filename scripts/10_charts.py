# -*- coding: utf-8 -*-
# 07~09번 결과 그래프: 모델별 PR-AUC, 처음 보는 설비 성능, 연속 경보 대기 시간별 오경보와 사건 재현율
# 입력: data/results/*.csv  출력: charts/model_pr_auc.png, charts/unit_split_pr_auc.png, charts/alarm_rule_tradeoff.png
#
# 사용법: python scripts/10_charts.py
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from evaluate import PROJECT_ROOT, RESULTS_DIR

# 02번 EDA 그래프와 같은 색. 세 번째 색(초록)은 배경 대비가 낮아 선마다 이름을 직접 붙임
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RANDOM_PR_AUC = 0.0292  # Validation 30분 양성 비율


def set_korean_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 110})


def model_pr_auc(res, out):
    order = ["규칙: 현재 전류고조파", "Logistic Regression (이력 없음 중앙값)", "Logistic Regression (이력 없음 표시)",
             "Random Forest", "LightGBM"]
    r = res.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(range(len(r)), r["pr_auc"], color=BLUE, height=0.55)
    for i, v in enumerate(r["pr_auc"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(RANDOM_PR_AUC, color=MUTED, ls="--", lw=1)
    ax.text(RANDOM_PR_AUC + 0.01, 0.5, "무작위 0.029", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(r)), r.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC (Validation)")
    ax.set_title("30분 내 이상 발생 예측 모델별 PR-AUC", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def unit_split(res, out):
    order = ["규칙: 현재 전류고조파", "Logistic Regression", "LightGBM"]
    p = res.pivot(index="model", columns="setting", values="pr_auc").loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    w = 0.36
    for j, (setting, color) in enumerate([("본 적 있는 설비", BLUE), ("처음 보는 설비", ORANGE)]):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(p))]
        ax.bar(xs, p[setting], width=w, color=color, label=setting)
        for x, v in zip(xs, p[setting]):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(p)), p.index, fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("PR-AUC")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("같은 설비의 뒤 기간 평가: 학습에 그 설비를 넣었을 때와 뺐을 때", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def alarm_tradeoff(res, out):
    names = {"규칙: 현재 전류고조파": ("규칙", ORANGE), "Logistic Regression (이력 없음 표시)": ("Logistic Regression", AQUA),
             "LightGBM": ("LightGBM", BLUE)}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for model, (label, color) in names.items():
        r = res[res["model"] == model].sort_values("wait_min")
        ax.plot(r["false_alarm_per_day"], r["event_recall"] * 100, color=color, lw=2, marker="o", ms=6, label=label)
        first = r.iloc[0]
        ax.annotate(label, (first["false_alarm_per_day"], first["event_recall"] * 100), xytext=(6, 4),
                    textcoords="offset points", fontsize=9, color=INK)
        if model == "LightGBM":
            for _, row in r.iterrows():
                ax.annotate(f"{int(row['wait_min'])}분", (row["false_alarm_per_day"], row["event_recall"] * 100),
                            xytext=(7, -12), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("설비 1대당 하루 오경보 (회)")
    ax.set_ylabel("사건 재현율 (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("경보 대기 시간(1~10분)에 따른 오경보와 놓친 사건", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="실험 결과 그래프")
    p.add_argument("--src", default=RESULTS_DIR)
    p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "charts"))
    args = p.parse_args()
    set_korean_font()
    model_pr_auc(pd.read_csv(os.path.join(args.src, "model_metrics.csv")), os.path.join(args.dst, "model_pr_auc.png"))
    alarm_tradeoff(pd.read_csv(os.path.join(args.src, "alarm_rule_metrics.csv")),
                   os.path.join(args.dst, "alarm_rule_tradeoff.png"))
    unit_path = os.path.join(args.src, "unit_split_metrics.csv")
    if os.path.exists(unit_path):
        unit_split(pd.read_csv(unit_path), os.path.join(args.dst, "unit_split_pr_auc.png"))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
