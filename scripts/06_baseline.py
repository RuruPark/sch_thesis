# -*- coding: utf-8 -*-
# 30분 내 이상 발생 예측 베이스라인 비교: 항상 음성, 전류고조파 임계값 규칙, Logistic Regression
# 입력: data/features  출력: data/results/baseline_metrics.csv, baseline_by_facility.csv
# Training으로 학습과 경보 임계값 선택, Validation(같은 설비의 뒤 기간)으로 평가
#
# 사용법: python scripts/06_baseline.py
import argparse, os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import FEATURES_DIR, RESULTS_DIR, TARGET, alarm_metrics, best_threshold, feature_columns, load


def main():
    p = argparse.ArgumentParser(description="베이스라인 비교")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    tr, va = load("Training", args.src), load("Validation", args.src)
    feats = feature_columns(tr)
    y_tr, y_va = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    print(f"Training {len(tr):,}행 양성 {y_tr.mean():.4f} / Validation {len(va):,}행 양성 {y_va.mean():.4f}")

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000))
    X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
    lr.fit(X_tr, y_tr)

    # 규칙 점수가 비어 있으면 경보를 낼 근거가 없으므로 가장 낮은 값으로 둠
    scores = {
        "항상 음성": (np.zeros(len(tr)), np.zeros(len(va))),
        "규칙: 현재 전류고조파": (tr["전류고조파평균"].fillna(-1).to_numpy(), va["전류고조파평균"].fillna(-1).to_numpy()),
        "규칙: 최근 10분 전류고조파 최댓값": (tr["전류고조파평균_max10"].fillna(-1).to_numpy(), va["전류고조파평균_max10"].fillna(-1).to_numpy()),
        "Logistic Regression": (lr.predict_proba(X_tr)[:, 1],
                                lr.predict_proba(va[feats].replace([np.inf, -np.inf], np.nan))[:, 1]),
    }

    rows, by_fac = [], []
    for name, (s_tr, s_va) in scores.items():
        if name == "항상 음성":
            thr, alarm = np.nan, np.zeros(len(va), dtype=bool)
        else:
            thr = best_threshold(y_tr, s_tr)
            alarm = s_va >= thr
        r = dict(model=name, threshold=thr, pr_auc=average_precision_score(y_va, s_va), **alarm_metrics(va, alarm))
        rows.append(r)
        for fac, idx in va.groupby("facility").indices.items():
            by_fac.append(dict(model=name, facility=fac, n=len(idx), positives=int(y_va[idx].sum()),
                               pr_auc=average_precision_score(y_va[idx], s_va[idx]) if y_va[idx].sum() else np.nan))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dst, "baseline_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(by_fac).to_csv(os.path.join(args.dst, "baseline_by_facility.csv"), index=False, encoding="utf-8-sig")

    coef = pd.Series(lr[-1].coef_[0], index=feats)
    coef.reindex(coef.abs().sort_values(ascending=False).index).to_csv(
        os.path.join(args.dst, "baseline_lr_coef.csv"), header=["coef"], encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
