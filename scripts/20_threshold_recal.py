# -*- coding: utf-8 -*-
# 새 설비의 처음 며칠로 경보 임계값만 다시 맞추면 처음 보는 설비 오경보가 줄어드는지
# 입력: 17번 --save-scores의 extra_fac_<태그>_scores.parquet, _thresholds.csv  출력: data/results/recal_<태그>.csv
# 처음 보는 설비 하루 오경보가 같은 설비 평가의 두 배 넘게 나옴. 임계값을 학습 설비 점수 분포로 고른 탓에 새 설비 분포와 어긋나는 문제 있음
# 모델은 다시 학습하지 않음. 설비마다 Validation 앞 N일을 보정 기간으로 쓰고, 지표는 모든 방식을 그 뒤 기간에서만 계산
#
# 사용법: python scripts/20_threshold_recal.py --scores data/results/extra_fac_v2cmp_scores.parquet [--days 1 3 7] [--tag v2cmp]
import argparse, os
import numpy as np
import pandas as pd

from evaluate import RESULTS_DIR, TARGET, alarm_metrics, best_threshold, sustained

MODELS = ["LightGBM 펌프+추가", "LightGBM 펌프만", "규칙"]
QUANTILES = [0.95, 0.98, 0.99]
WAITS = [1, 2]


def unit_thresholds(d, col, base, days, qs, target):
    # 보정 기간: 그 설비의 Validation 첫 시점부터 N일. 운전 시점만 남아 있어 행 수는 설비마다 다름
    start = d.groupby("dev")["time"].transform("min")
    calib = (d["time"] < start + pd.Timedelta(days=days)).to_numpy()
    thr = {"기존": d["fold"].map(base).to_numpy()}
    # 라벨 사용: 보정 기간에 이상 발생이 있던 설비만 F1 최대 지점으로 다시 고름. 없으면 기존 임계값 유지
    lab = thr["기존"].copy()
    for q in qs:
        thr[f"점수 {int(q * 100)}% 분위 상한"] = thr["기존"].copy()
    for dev, idx in d.groupby("dev").indices.items():
        c = idx[calib[idx]]
        if len(c) == 0:
            continue
        y, s = d[target].to_numpy()[c], d[col].to_numpy()[c]
        if y.sum() > 0:
            lab[idx] = best_threshold(y, s)
        # 라벨 없이: 보정 기간 점수의 상위 (1-q)만 경보가 되도록 하한을 올림. 기존보다 낮추지는 않아 경보가 늘지 않음
        for q in qs:
            name = f"점수 {int(q * 100)}% 분위 상한"
            thr[name][idx] = np.maximum(thr["기존"][idx], np.quantile(s, q))
    thr["라벨 사용 F1"] = lab
    return thr, calib


def main():
    p = argparse.ArgumentParser(description="새 설비 처음 며칠로 임계값 재보정")
    p.add_argument("--scores", required=True)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--days", type=float, nargs="*", default=[1, 3, 7])
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    sc = pd.read_parquet(args.scores)
    th = pd.read_csv(args.scores.replace("_scores.parquet", "_thresholds.csv"))
    sc["time"] = pd.to_datetime(sc["time"])
    rows = []
    for model in args.models:
        t = th[th["model"] == model]
        if t.empty:
            continue
        col = t["column"].iloc[0]
        base = t.set_index("fold")["threshold"]
        for days in args.days:
            thr, calib = unit_thresholds(sc, col, base, days, QUANTILES, args.target)
            ev = sc[~calib].set_index("time")
            s = sc[col].to_numpy()[~calib]
            for name, v in thr.items():
                raw = s >= v[~calib]
                for w in WAITS:
                    rows.append(dict(model=model, days=days, method=name, wait_min=w, eval_units=ev["dev"].nunique(),
                                     **alarm_metrics(ev, sustained(ev, raw, w), args.target)))
            print(f"{model} {days}일 완료")
    res = pd.DataFrame(rows)
    path = os.path.join(args.dst, "recal" + (f"_{args.tag}" if args.tag else "") + ".csv")
    res.to_csv(path, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
        print(res[["model", "days", "method", "wait_min", "precision", "f1", "event_recall", "lead_median",
                   "false_alarm_per_day"]].round(3).to_string(index=False))
    print("완료:", path)


if __name__ == "__main__":
    main()
