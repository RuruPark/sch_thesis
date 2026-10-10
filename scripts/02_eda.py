# -*- coding: utf-8 -*-
# 1차 전처리 결과(설비별 parquet 210개) 전체를 보고 결측, 이상값, 라벨 분포를 정리
# 입력: data/processed  출력: data/eda/*.csv, charts/eda_*.png
# 2차 전처리(03_clean.py)의 기준값은 여기서 나온 표를 근거로 정함
#
# 사용법: python scripts/02_eda.py
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE_I = ["R상전류", "S상전류", "T상전류"]
PHASE_V = ["R상전압", "S상전압", "T상전압"]
STATE = "label_기동_전류평균"
SOH = {"label_SOH_역률평균": "역률평균",
       "label_SOH_전류고조파평균": "전류고조파평균",
       "label_SOH_전압고조파평균": "전압고조파평균"}
# 라벨별 2~98% 구간을 보고 잡은 경계. 운전 구간에서 이 경계로 라벨이 거의 그대로 재현됨
SOH_CUTS = {"역률평균": (0.8, 0.6), "전류고조파평균": (30, 60), "전압고조파평균": (3, 5)}

GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def set_korean_font():
    # 윈도우는 맑은 고딕, 리눅스는 설치된 CJK 폰트 중 있는 것
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 110})


def load_all(src):
    out = []
    for f in sorted(glob.glob(os.path.join(src, "*", "*", "*.parquet"))):
        split, facility = f.split(os.sep)[-3:-1]
        d = pd.read_parquet(f)
        num = d.select_dtypes("number").columns
        d[num] = d[num].astype("float32")  # float64로 다 올리면 2GB 넘어서 줄임
        d = d.reset_index()
        d["split"], d["facility"], d["dev"] = split, facility, int(os.path.basename(f)[:-8])
        out.append(d)
    return pd.concat(out, ignore_index=True)


def soh_rule(col, x):
    hi, lo = SOH_CUTS[col]
    if col == "역률평균":
        return np.select([x >= hi, x >= lo], ["정상", "주의"], "경고")
    return np.select([x < hi, x < lo], ["정상", "주의"], "경고")


def summarize(df, out_dir):
    key = ["split", "facility", "dev"]
    num = [c for c in df.select_dtypes("number").columns if c != "dev"]
    g = df.groupby(key)
    dt = g["TIMESTAMP"].diff().dt.total_seconds()
    # 299~301번은 꺼진 동안 상전압 7V가 찍혀서 0이 아니라 50V 미만 기준
    power_off = df["상전압평균"] < 50
    clip = (df[PHASE_I] == 512).any(axis=1) | (df[PHASE_I] >= 4095).any(axis=1)
    dev = pd.DataFrame({
        "n_rows": g.size(),
        "start": g["TIMESTAMP"].min(), "end": g["TIMESTAMP"].max(),
        "gap_over_5min": (dt > 300).groupby([df[k] for k in key]).sum(),
        "missing_min": ((dt[dt > 90] / 60).round() - 1).groupby([df[k] for k in key]).sum(),
        "nan_rows": df[num].isna().any(axis=1).groupby([df[k] for k in key]).sum(),
        "power_off_ratio": power_off.groupby([df[k] for k in key]).mean(),
        "stop_ratio": (df[STATE] == "Stop").groupby([df[k] for k in key]).mean(),
        "neg_power_ratio": (df["유효전력평균"] < 0).groupby([df[k] for k in key]).mean(),
        "neg_energy_ratio": (df["누적전력량"] < 0).groupby([df[k] for k in key]).mean(),
        "freq_out_ratio": (~power_off & df["주파수"].notna() & ~df["주파수"].between(55, 65)).groupby([df[k] for k in key]).mean(),
        "clip_rows": clip.groupby([df[k] for k in key]).sum(),
        "volt_spike_rows": (df[PHASE_V] > 500).any(axis=1).groupby([df[k] for k in key]).sum(),
    })
    dev.round({c: 4 for c in dev.columns if dev[c].dtype.kind == "f"}).to_csv(os.path.join(out_dir, "device_summary.csv"), encoding="utf-8-sig")

    col = df[num].describe(percentiles=[.001, .01, .5, .99, .999]).T
    col["nan"] = df[num].isna().sum()
    col["zero_ratio"] = (df[num] == 0).mean()
    col["neg_ratio"] = (df[num] < 0).mean()
    col.round(4).to_csv(os.path.join(out_dir, "column_summary.csv"), encoding="utf-8-sig")

    labels = [c for c in df.columns if c.startswith("label_")]
    rows = []
    for c in labels:
        for (sp, fa), s in df.groupby(["split", "facility"])[c]:
            for v, n in s.value_counts().items():
                rows.append(dict(label=c, split=sp, facility=fa, value=v, n=n, ratio=n / len(s)))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out_dir, "label_distribution.csv"),
                                       index=False, encoding="utf-8-sig")

    by_state = pd.concat({c: pd.crosstab(df[STATE], df[c], normalize="index") for c in SOH}, axis=1)
    by_state.round(4).to_csv(os.path.join(out_dir, "soh_by_state.csv"), encoding="utf-8-sig")

    run = df[df[STATE] != "Stop"]
    match = {c: (soh_rule(v, run[v].to_numpy()) == run[c].to_numpy()).mean() for c, v in SOH.items()}
    pd.Series(match, name="rule_match").round(4).to_csv(os.path.join(out_dir, "soh_rule_match.csv"),
                                                        encoding="utf-8-sig")
    return dev, by_state, match


def chart_soh_by_state(by_state, path):
    t = by_state["label_SOH_역률평균"].reindex(["Stop", "Unloading", "Loading"])[["정상", "주의", "경고"]]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    left = np.zeros(len(t))
    for lab, color in zip(t.columns, [GOOD, WARN, CRIT]):
        ax.barh(t.index, t[lab], left=left, color=color, edgecolor="white", linewidth=2, height=0.6, label=lab)
        for y, (l, w) in enumerate(zip(left, t[lab])):
            if w > 0.08:
                ax.text(l + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=9, color=INK)
        left += t[lab].to_numpy()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("기동 상태별 SOH(역률) 라벨 비율", loc="left", fontsize=11, color=INK)
    ax.legend(ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_soh_thresholds(df, path):
    run = df[df[STATE] != "Stop"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ranges = {"역률평균": (-0.2, 1.05), "전류고조파평균": (0, 130), "전압고조파평균": (0, 12)}
    for ax, (lab, c) in zip(axes, SOH.items()):
        bins = np.linspace(*ranges[c], 80)
        for v, color in zip(["정상", "주의", "경고"], [GOOD, WARN, CRIT]):
            ax.hist(run.loc[run[lab] == v, c], bins=bins, color=color, alpha=0.75, label=v)
        for x in SOH_CUTS[c]:
            ax.axvline(x, color=INK, linewidth=1, linestyle="--")
        ax.set_title(c, loc="left", fontsize=10, color=INK)
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("운전 구간의 SOH 라벨별 기준 항목 분포 (점선: 추정 경계)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_clip_example(df, path):
    d = df[(df.split == "Training") & (df.dev == 484)].set_index("TIMESTAMP")
    d = d.loc["2021-02-04 18:30":"2021-02-04 19:30"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(d.index, d["전류평균"], color=BLUE, linewidth=2)
    hit = d[(d[PHASE_I] == 512).any(axis=1)]
    ax.scatter(hit.index, hit["전류평균"], s=40, color=CRIT, zorder=3, edgecolor="white", linewidth=2)
    for t, v in hit["전류평균"].items():
        ax.annotate("512 A (측정 상한)", (t, v), xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_ylabel("전류평균 (A)")
    ax.set_title("기동 순간 전류가 512 A로 잘려 기록된 예 (펌프 484번)", loc="left", fontsize=11, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_device_ratio(dev, path):
    d = dev.loc["Training"].sort_values("stop_ratio")
    fig, ax = plt.subplots(figsize=(9, 3))
    x = np.arange(len(d))
    ax.bar(x, d["stop_ratio"], color=BLUE, width=0.8, label="정지(Stop) 비율")
    ax.bar(x, d["power_off_ratio"], color=ORANGE, width=0.8, label="전원 꺼짐 비율")
    ax.set_xticks([])
    ax.set_xlabel("설비 105대 (Training, 정지 비율 순)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("설비별 정지 구간 비율", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="1차 전처리 결과 EDA")
    p.add_argument("--src", default=os.path.join(PROJECT_ROOT, "data", "processed"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "eda"))
    p.add_argument("--charts", default=os.path.join(PROJECT_ROOT, "charts"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.charts, exist_ok=True)
    set_korean_font()

    df = load_all(args.src)
    print(f"{df.groupby(['split', 'dev']).ngroups}개 파일, {len(df):,}행")
    dev, by_state, match = summarize(df, args.out)
    print("SOH 라벨 경계 재현율(운전 구간):", {k: round(v, 3) for k, v in match.items()})

    chart_device_ratio(dev, os.path.join(args.charts, "eda_stop_ratio.png"))
    chart_soh_by_state(by_state, os.path.join(args.charts, "eda_soh_pf_by_state.png"))
    chart_soh_thresholds(df, os.path.join(args.charts, "eda_soh_thresholds.png"))
    chart_clip_example(df, os.path.join(args.charts, "eda_current_clip.png"))
    print("완료:", args.out, args.charts)


if __name__ == "__main__":
    main()
