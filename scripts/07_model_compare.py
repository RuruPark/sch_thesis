# -*- coding: utf-8 -*-
# 30분 내 이상 발생 예측 모델 비교: 전류고조파 규칙, Logistic Regression, Random Forest, LightGBM
# 입력: data/features  출력: data/results/model_metrics.csv, model_scores.parquet, lgbm_importance.csv
# 설비마다 Training 앞 80% 기간으로 학습, 뒤 20%로 경보 임계값과 조기 종료 기준을 정하고 Validation으로 평가
#
# 사용법: python scripts/07_model_compare.py
import argparse, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (FEATURES_DIR, HIST, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics, best_threshold,
                      clean, feature_columns, load, split_fit_calib)


def main():
    p = argparse.ArgumentParser(description="모델 비교")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    fit, cal = split_fit_calib(load("Training", args.src))
    va = load("Validation", args.src)
    feats = feature_columns(fit)
    hist_fill = float(fit[HIST].max())
    fit_b, cal_b, va_b = (add_history_flag(x, hist_fill) for x in (fit, cal, va))
    feats_b = feats + ["이상이력없음"]
    y_fit, y_cal = fit[TARGET].to_numpy(), cal[TARGET].to_numpy()
    print(f"학습 {len(fit):,}행, 임계값용 {len(cal):,}행, Validation {len(va):,}행")

    def lr():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))

    scores = {}
    rule = lambda x: x["전류고조파평균"].fillna(-1).to_numpy()
    scores["규칙: 현재 전류고조파"] = (rule(cal), rule(va))

    m = lr().fit(clean(fit[feats]), y_fit)
    scores["Logistic Regression (이력 없음 중앙값)"] = (m.predict_proba(clean(cal[feats]))[:, 1],
                                                    m.predict_proba(clean(va[feats]))[:, 1])
    m = lr().fit(clean(fit_b[feats_b]), y_fit)
    scores["Logistic Regression (이력 없음 표시)"] = (m.predict_proba(clean(cal_b[feats_b]))[:, 1],
                                                   m.predict_proba(clean(va_b[feats_b]))[:, 1])
    print("Logistic Regression 완료")

    # 트리 200개, 트리마다 학습 행의 10%만 씀. 전체 행으로 키울 경우, 4코어에서 한 시간 넘게 걸리는 문제 있음
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=200, max_samples=0.1, min_samples_leaf=20,
                                              class_weight="balanced_subsample", n_jobs=-1, random_state=0))
    rf.fit(clean(fit_b[feats_b]), y_fit)
    scores["Random Forest"] = (rf.predict_proba(clean(cal_b[feats_b]))[:, 1], rf.predict_proba(clean(va_b[feats_b]))[:, 1])
    print("Random Forest 완료")

    gbm = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    gbm.fit(clean(fit_b[feats_b]), y_fit, eval_set=[(clean(cal_b[feats_b]), y_cal)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    scores["LightGBM"] = (gbm.predict_proba(clean(cal_b[feats_b]))[:, 1], gbm.predict_proba(clean(va_b[feats_b]))[:, 1])
    print(f"LightGBM 완료 (트리 {gbm.best_iteration_}개)")

    y_va = va[TARGET].to_numpy()
    rows = []
    for name, (s_cal, s_va) in scores.items():
        thr = best_threshold(y_cal, s_cal)
        rows.append(dict(model=name, threshold=thr, pr_auc_calib=average_precision_score(y_cal, s_cal),
                         pr_auc=average_precision_score(y_va, s_va), **alarm_metrics(va, s_va >= thr)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dst, "model_metrics.csv"), index=False, encoding="utf-8-sig")

    out = va[["dev", "facility", TARGET, "minutes_to_event"]].copy()
    for name, (_, s_va) in scores.items():
        out[name] = s_va.astype("float32")
    out.to_parquet(os.path.join(args.dst, "model_scores.parquet"))
    pd.DataFrame([dict(model=r["model"], threshold=r["threshold"]) for r in rows]).to_csv(
        os.path.join(args.dst, "model_thresholds.csv"), index=False, encoding="utf-8-sig")
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=feats_b)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(
        os.path.join(args.dst, "lgbm_importance.csv"), header=["gain_share"], encoding="utf-8-sig")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
