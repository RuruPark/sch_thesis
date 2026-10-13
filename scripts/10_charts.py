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


def model_pr_auc(res, out, random_pr_auc):
    order = ["규칙: 현재 전류고조파", "Logistic Regression (이력 없음 중앙값)", "Logistic Regression (이력 없음 표시)",
             "Random Forest", "LightGBM"]
    r = res.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(range(len(r)), r["pr_auc"], color=BLUE, height=0.55)
    for i, v in enumerate(r["pr_auc"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    # 무작위 점수의 PR-AUC는 양성 비율과 같음
    ax.axvline(random_pr_auc, color=MUTED, ls="--", lw=1)
    ax.text(random_pr_auc + 0.01, 0.5, f"무작위 {random_pr_auc:.3f}", va="center", fontsize=8, color=MUTED)
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
    order = ["규칙", "Logistic Regression", "LightGBM"]
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


def shap_top(imp, out, n=15):
    r = imp.head(n)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(range(len(r)), r["mean_abs_shap"], color=BLUE, height=0.6)
    for i, (v, sh) in enumerate(zip(r["mean_abs_shap"], r["share"])):
        ax.text(v + r["mean_abs_shap"].max() * 0.01, i, f"{sh:.1%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(r)), r["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("평균 |SHAP| (log-odds)")
    ax.set_title(f"LightGBM 예측에 크게 기여한 특징 상위 {n}개", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def shap_dependence(d, out):
    feature = d["feature"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for t, color, label in [(0, BLUE, "30분 안에 이상 없음"), (1, ORANGE, "30분 안에 이상 발생")]:
        r = d[d["target"] == t]
        ax.scatter(r["value"], r["shap"], s=6, alpha=0.35 if t == 0 else 0.6, color=color, label=label, linewidths=0)
    ax.axhline(0, color=MUTED, lw=1)
    if d["value"].max() > 1000:
        # 이상 이력 없음은 학습 기간 최댓값으로 채워 오른쪽 끝에 몰림. 선형 축일 경우, 나머지 점이 왼쪽에 뭉치는 문제 있음
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.annotate("이상 이력 없음", (d["value"].max(), 0), xytext=(-8, 10), textcoords="offset points",
                    ha="right", fontsize=8, color=MUTED)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP (log-odds)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", markerscale=3)
    ax.set_title(f"{feature} 값에 따른 예측 기여", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="실험 결과 그래프")
    p.add_argument("--src", default=RESULTS_DIR)
    p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "charts"))
    args = p.parse_args()
    set_korean_font()
    base = pd.read_csv(os.path.join(args.src, "baseline_metrics.csv")).set_index("model")
    model_pr_auc(pd.read_csv(os.path.join(args.src, "model_metrics.csv")), os.path.join(args.dst, "model_pr_auc.png"),
                 float(base.loc["항상 음성", "pr_auc"]))
    alarm_tradeoff(pd.read_csv(os.path.join(args.src, "alarm_rule_metrics.csv")),
                   os.path.join(args.dst, "alarm_rule_tradeoff.png"))
    unit_path = os.path.join(args.src, "unit_split_metrics.csv")
    if os.path.exists(unit_path):
        unit_split(pd.read_csv(unit_path), os.path.join(args.dst, "unit_split_pr_auc.png"))
    shap_path = os.path.join(args.src, "shap_importance.csv")
    if os.path.exists(shap_path):
        shap_top(pd.read_csv(shap_path), os.path.join(args.dst, "shap_top.png"))
        shap_dependence(pd.read_parquet(os.path.join(args.src, "shap_dependence_sample.parquet")),
                        os.path.join(args.dst, "shap_dependence.png"))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
