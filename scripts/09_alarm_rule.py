# -*- coding: utf-8 -*-
# 연속 경보 규칙: 경보가 N분 이어질 때만 알리면 오경보와 놓친 사건이 어떻게 바뀌는지
# 입력: data/results/model_scores.parquet, model_thresholds.csv (07번 결과)  출력: data/results/alarm_rule_metrics.csv
#
# 사용법: python scripts/09_alarm_rule.py
import argparse, os
import pandas as pd

from evaluate import RESULTS_DIR, alarm_metrics, sustained

MODELS = ["규칙: 현재 전류고조파", "Logistic Regression (이력 없음 표시)", "LightGBM"]
# 30분 예측이라 10분 넘게 기다릴 경우, 남는 대응 시간이 너무 짧아지는 문제 있어 10분까지만 봄
WAIT_MINUTES = [1, 2, 3, 5, 10]


def main():
    p = argparse.ArgumentParser(description="연속 경보 규칙 비교")
    p.add_argument("--dir", default=RESULTS_DIR)
    args = p.parse_args()

    va = pd.read_parquet(os.path.join(args.dir, "model_scores.parquet"))
    thr = pd.read_csv(os.path.join(args.dir, "model_thresholds.csv")).set_index("model")["threshold"]

    rows = []
    for name in MODELS:
        raw = (va[name] >= thr[name]).to_numpy()
        for n in WAIT_MINUTES:
            rows.append(dict(model=name, wait_min=n, **alarm_metrics(va, sustained(va, raw, n))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dir, "alarm_rule_metrics.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print("완료:", args.dir)


if __name__ == "__main__":
    main()
