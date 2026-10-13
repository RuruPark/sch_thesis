# -*- coding: utf-8 -*-
# 처음 보는 설비에 대한 성능: 설비를 5묶음으로 나눠, 한 묶음을 빼고 학습한 모델과 전체로 학습한 모델을 같은 설비에서 비교
# 입력: data/features  출력: data/results/unit_split{_태그}_metrics.csv, _folds.csv, _assign.csv
# 평가는 두 경우 모두 빠진 묶음 설비의 Validation 기간. 차이는 그 설비의 Training 기간을 학습에 넣었는지뿐
# 예측 구간(--target), 설비별 평소 값 대비 특징(--norm), 다른 라벨의 특징 폴더(--src)를 바꿔 같은 틀로 비교
#
# 사용법: python scripts/08_unit_split.py [--target target_10] [--norm] [--src data/features_vh --rule-col 전압고조파평균] [--tag 이름]
import argparse, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (FEATURES_DIR, HIST, RESULTS_DIR, TARGET, add_history_flag, add_unit_norm, alarm_metrics,
                      best_threshold, clean, feature_columns, load, split_fit_calib)

N_FOLDS = 5


def assign_folds(tr, target):
    # 양성이 있는 설비가 적어, 무작위로 나눌 경우 한 묶음에 사건 많은 설비가 몰리는 문제 있어 양성 수 순서대로 돌아가며 배정
    pos = tr.groupby("dev")[target].sum().sort_values(ascending=False, kind="stable")
    with_pos, without = pos[pos > 0].index, pos[pos == 0].index
    folds = {d: i % N_FOLDS for i, d in enumerate(with_pos)}
    folds.update({d: i % N_FOLDS for i, d in enumerate(without)})
    return folds


def lgbm():
    return lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def train_models(tr, target, rule_col):
    fit, cal = split_fit_calib(tr)
    fill = float(fit[HIST].max())
    fit, cal = add_history_flag(fit, fill), add_history_flag(cal, fill)
    feats = feature_columns(fit)
    y_fit, y_cal = fit[target].to_numpy(), cal[target].to_numpy()

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000)).fit(clean(fit[feats]), y_fit)
    # 07번과 같은 설정. 조기 종료 기준은 학습 설비의 뒤 20% 기간
    gbm = lgbm().fit(clean(fit[feats]), y_fit, eval_set=[(clean(cal[feats]), y_cal)], eval_metric="average_precision",
                     callbacks=[lgb.early_stopping(100, verbose=False)])

    models = {
        "규칙": lambda x: x[rule_col].fillna(-1).to_numpy(),
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
    p.add_argument("--target", default=TARGET)
    p.add_argument("--rule-col", default="전류고조파평균")
    p.add_argument("--norm", action="store_true", help="설비별 평소 값 대비 특징 추가")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    target = args.target
    prefix = os.path.join(args.dst, "unit_split" + (f"_{args.tag}" if args.tag else ""))

    tr, va = load("Training", args.src, target), load("Validation", args.src, target)
    if args.norm:
        tr, va = add_unit_norm(tr, va)
    folds = assign_folds(tr, target)
    tr_fold, va_fold = tr["dev"].map(folds), va["dev"].map(folds)
    print(f"{target}: Training {len(tr):,}행, Validation {len(va):,}행, 특징 {len(feature_columns(tr)) + 1}개")

    seen, seen_fill = train_models(tr, target, args.rule_col)
    print("전체 설비 모델 완료")

    pooled, fold_rows = {}, []
    for k in range(N_FOLDS):
        te = va[va_fold == k]
        unseen, unseen_fill = train_models(tr[tr_fold != k], target, args.rule_col)
        y = te[target].to_numpy()
        for setting, (models, fill) in {"본 적 있는 설비": (seen, seen_fill), "처음 보는 설비": (unseen, unseen_fill)}.items():
            for name, (s, thr) in score(models, fill, te).items():
                pooled.setdefault((setting, name), []).append((s, s >= thr))
                fold_rows.append(dict(fold=k, setting=setting, model=name, units=te["dev"].nunique(),
                                      positives=int(y.sum()), pr_auc=average_precision_score(y, s) if y.sum() else np.nan))
        print(f"묶음 {k} 완료: 설비 {te['dev'].nunique()}대, 양성 {int(y.sum()):,}")

    # 묶음별 임계값으로 낸 경보를 모아 전체 설비 기준 지표 계산. 행 순서를 묶음 순서로 맞춤
    va_sorted = pd.concat([va[va_fold == k] for k in range(N_FOLDS)])
    y_all = va_sorted[target].to_numpy()
    rows = []
    for (setting, name), parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        rows.append(dict(setting=setting, model=name, positive_rate=y_all.mean(), pr_auc=average_precision_score(y_all, s),
                         **alarm_metrics(va_sorted, alarm, target)))
    res = pd.DataFrame(rows)
    res.to_csv(prefix + "_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(prefix + "_folds.csv", index=False, encoding="utf-8-sig")
    pd.Series(folds, name="fold").rename_axis("dev").to_csv(prefix + "_assign.csv")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index=["setting", "model"], columns="fold", values="pr_auc").round(3))
    print("완료:", prefix)


if __name__ == "__main__":
    main()
