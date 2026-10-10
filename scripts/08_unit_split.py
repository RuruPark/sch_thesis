# -*- coding: utf-8 -*-
# 처음 보는 설비에 대한 성능: 설비를 5묶음으로 나눠, 한 묶음을 빼고 학습한 모델과 전체로 학습한 모델을 같은 설비에서 비교
# 입력: data/features  출력: data/results/unit_split_metrics.csv, unit_split_folds.csv
# 평가는 두 경우 모두 빠진 묶음 설비의 Validation 기간. 차이는 그 설비의 Training 기간을 학습에 넣었는지뿐
#
# 사용법: python scripts/08_unit_split.py
import argparse, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (FEATURES_DIR, HIST, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics, best_threshold,
                      clean, feature_columns, load, split_fit_calib)

N_FOLDS = 5


def assign_folds(tr):
    # 양성이 있는 설비가 105대 중 35대뿐이라, 무작위로 나눌 경우 한 묶음에 사건 많은 설비가 몰리는 문제 있어 양성 수 순서대로 돌아가며 배정
    pos = tr.groupby("dev")[TARGET].sum().sort_values(ascending=False, kind="stable")
    with_pos, without = pos[pos > 0].index, pos[pos == 0].index
    folds = {d: i % N_FOLDS for i, d in enumerate(with_pos)}
    folds.update({d: i % N_FOLDS for i, d in enumerate(without)})
    return folds


def train_models(tr):
    fit, cal = split_fit_calib(tr)
    fill = float(fit[HIST].max())
    fit, cal = add_history_flag(fit, fill), add_history_flag(cal, fill)
    feats = feature_columns(fit)
    y_fit, y_cal = fit[TARGET].to_numpy(), cal[TARGET].to_numpy()

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000)).fit(clean(fit[feats]), y_fit)
    # 07번과 같은 설정. 조기 종료 기준은 학습 설비의 뒤 20% 기간
    gbm = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    gbm.fit(clean(fit[feats]), y_fit, eval_set=[(clean(cal[feats]), y_cal)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)])

    models = {
        "규칙: 현재 전류고조파": lambda x: x["전류고조파평균"].fillna(-1).to_numpy(),
        "Logistic Regression": lambda x: lr.predict_proba(clean(x[feats]))[:, 1],
        "LightGBM": lambda x: gbm.predict_proba(clean(x[feats]))[:, 1],
    }
    return {name: (f, best_threshold(y_cal, f(cal))) for name, f in models.items()}, fill


def score(models, fill, te):
    te = add_history_flag(te, fill)
    return {name: (f(te), thr) for name, (f, thr) in models.items()}


def main():
    p = argparse.ArgumentParser(description="설비 단위 분할 평가")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    tr, va = load("Training", args.src), load("Validation", args.src)
    folds = assign_folds(tr)
    tr_fold, va_fold = tr["dev"].map(folds), va["dev"].map(folds)

    seen, seen_fill = train_models(tr)
    print("전체 설비 모델 완료")

    pooled = {}
    fold_rows = []
    for k in range(N_FOLDS):
        te = va[va_fold == k]
        unseen, unseen_fill = train_models(tr[tr_fold != k])
        for setting, (models, fill) in {"본 적 있는 설비": (seen, seen_fill), "처음 보는 설비": (unseen, unseen_fill)}.items():
            for name, (s, thr) in score(models, fill, te).items():
                pooled.setdefault((setting, name), []).append((s, s >= thr))
                y = te[TARGET].to_numpy()
                fold_rows.append(dict(fold=k, setting=setting, model=name, units=te["dev"].nunique(),
                                      positives=int(y.sum()), pr_auc=average_precision_score(y, s) if y.sum() else np.nan))
        print(f"묶음 {k} 완료: 설비 {te['dev'].nunique()}대, 양성 {int(te[TARGET].sum()):,}")

    # 묶음별 임계값으로 낸 경보를 모아 전체 설비 기준 지표 계산. 행 순서를 묶음 순서로 맞춤
    va_sorted = pd.concat([va[va_fold == k] for k in range(N_FOLDS)])
    y_all = va_sorted[TARGET].to_numpy()
    rows = []
    for (setting, name), parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        rows.append(dict(setting=setting, model=name, pr_auc=average_precision_score(y_all, s),
                         **alarm_metrics(va_sorted, alarm)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dst, "unit_split_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.dst, "unit_split_folds.csv"), index=False, encoding="utf-8-sig")
    pd.Series(folds, name="fold").rename_axis("dev").to_csv(os.path.join(args.dst, "unit_split_assign.csv"))

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index=["setting", "model"], columns="fold", values="pr_auc").round(3))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
