# -*- coding: utf-8 -*-
# LightGBM 하이퍼파라미터 탐색: Training 안에서만 처음 보는 설비 성능으로 설정을 고르고, 고른 설정으로 08번과 같은 평가
# 입력: data/features, data/results/unit_split_assign.csv  출력: data/results/tune_search.csv, tune_metrics.csv, tune_folds.csv
# Validation으로 설정을 고를 경우, 처음 보는 설비 성능이 부풀려지는 문제 있어 Validation은 마지막 평가에만 씀
#
# 사용법: python scripts/15_tune.py [--trials 20]
import argparse, json, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from evaluate import (HIST, LGBM_PARAMS, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics, best_threshold, clean,
                      feature_columns, load, split_fit_calib)

# 처음 보는 설비에서 떨어지는 폭이 커서, 잎 수를 줄이고 잎당 최소 행을 늘리는 쪽까지 넓게 봄
SPACE = dict(num_leaves=[15, 31, 63, 127], min_child_samples=[50, 100, 300, 1000], learning_rate=[0.03, 0.05, 0.1],
             colsample_bytree=[0.5, 0.8], subsample=[0.6, 0.8], reg_lambda=[0.0, 1.0, 10.0], scale_pos_weight=[1.0, 5.0])


def fit(params, fit_rows, cal_rows, feats):
    p = {**LGBM_PARAMS, **params}
    m = lgb.LGBMClassifier(**p).fit(clean(fit_rows[feats]), fit_rows[TARGET], eval_set=[(clean(cal_rows[feats]), cal_rows[TARGET])],
                                   eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def inner_score(params, fit_all, cal_all, fit_fold, cal_fold, feats, n_folds):
    # 묶음 k를 빼고 앞 80%로 학습, 남은 설비의 뒤 20%로 조기 종료, 묶음 k 설비의 뒤 20%로 채점
    ys, ss, trees = [], [], []
    for k in range(n_folds):
        m = fit(params, fit_all[fit_fold != k], cal_all[cal_fold != k], feats)
        te = cal_all[cal_fold == k]
        ys.append(te[TARGET].to_numpy())
        ss.append(m.predict_proba(clean(te[feats]))[:, 1])
        trees.append(m.best_iteration_)
    return average_precision_score(np.concatenate(ys), np.concatenate(ss)), int(np.median(trees))


def main():
    p = argparse.ArgumentParser(description="LightGBM 하이퍼파라미터 탐색")
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()

    tr, va = load("Training"), load("Validation")
    folds = pd.read_csv(os.path.join(args.dst, "unit_split_assign.csv")).set_index("dev")["fold"]
    n_folds = int(folds.max()) + 1
    fit_all, cal_all = split_fit_calib(tr)
    fill = float(fit_all[HIST].max())
    fit_all, cal_all = add_history_flag(fit_all, fill), add_history_flag(cal_all, fill)
    feats = feature_columns(fit_all)
    fit_fold, cal_fold = fit_all["dev"].map(folds).to_numpy(), cal_all["dev"].map(folds).to_numpy()

    # 0번은 07, 08번에서 쓴 기본 설정. 나머지는 고정 시드로 무작위 조합
    rng = np.random.default_rng(0)
    base = {k: LGBM_PARAMS[k] for k in ["num_leaves", "min_child_samples", "learning_rate", "colsample_bytree", "subsample"]}
    cands = [{**base, "reg_lambda": 0.0, "scale_pos_weight": 1.0}]
    while len(cands) < args.trials + 1:
        c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        if c not in cands:
            cands.append(c)

    rows = []
    for i, c in enumerate(cands):
        t0 = time.time()
        score, trees = inner_score(c, fit_all, cal_all, fit_fold, cal_fold, feats, n_folds)
        rows.append(dict(trial=i, inner_pr_auc=score, trees=trees, sec=round(time.time() - t0), **c))
        print(f"{i:2d} 내부 PR-AUC {score:.4f} 트리 {trees} {c}", flush=True)
    search = pd.DataFrame(rows).sort_values("inner_pr_auc", ascending=False)
    search.to_csv(os.path.join(args.dst, "tune_search.csv"), index=False, encoding="utf-8-sig")
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.iloc[0][list(SPACE)].items()}
    best["num_leaves"], best["min_child_samples"] = int(best["num_leaves"]), int(best["min_child_samples"])
    print("고른 설정:", best)
    with open(os.path.join(args.dst, "tune_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1)

    # 고른 설정으로 08번과 같은 평가. 임계값은 학습 설비의 뒤 20%에서 F1 최대
    va_b = add_history_flag(va, fill)
    va_fold = va["dev"].map(folds).to_numpy()

    def model_for(keep):
        m = fit(best, fit_all[keep(fit_fold)], cal_all[keep(cal_fold)], feats)
        cal = cal_all[keep(cal_fold)]
        return m, best_threshold(cal[TARGET].to_numpy(), m.predict_proba(clean(cal[feats]))[:, 1])

    seen = model_for(lambda f: np.ones(len(f), bool))
    pooled, fold_rows = {}, []
    for k in range(n_folds):
        # 이력 없음 채움값은 전체 학습 기간 최댓값 하나로 씀. 08번은 묶음마다 구하지만 실제 이력보다 크기만 하면 트리 분할은 같음
        unseen = model_for(lambda f, k=k: f != k)
        te = va_b[va_fold == k]
        y = te[TARGET].to_numpy()
        for setting, (m, thr) in {"본 적 있는 설비": seen, "처음 보는 설비": unseen}.items():
            s = m.predict_proba(clean(te[feats]))[:, 1]
            pooled.setdefault(setting, []).append((s, s >= thr))
            fold_rows.append(dict(fold=k, setting=setting, model="LightGBM (튜닝)", pr_auc=average_precision_score(y, s)))
        print(f"묶음 {k} 완료", flush=True)

    va_sorted = pd.concat([va_b[va_fold == k] for k in range(n_folds)])
    y_all = va_sorted[TARGET].to_numpy()
    res = []
    for setting, parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        res.append(dict(setting=setting, model="LightGBM (튜닝)", positive_rate=y_all.mean(),
                        pr_auc=average_precision_score(y_all, s), **alarm_metrics(va_sorted, alarm)))
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(args.dst, "tune_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.dst, "tune_folds.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(search.head(10).round(4).to_string(index=False))
        print(res.round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index="setting", columns="fold", values="pr_auc").round(3))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
