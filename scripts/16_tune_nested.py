# -*- coding: utf-8 -*-
# 15번 튜닝 결과 검증: 바깥 묶음마다 그 묶음 설비를 아예 빼고 남은 4묶음 안에서만 설정을 다시 고름 (중첩 교차검증)
# 입력: data/features, data/results/unit_split_assign.csv, tune_search.csv (후보 설정 목록만 씀)
# 출력: data/results/tune_nested_metrics.csv, tune_nested_folds.csv
# 15번은 평가 묶음 설비의 Training 뒤 20%로 설정을 골라, 같은 설비의 Validation 점수가 조금 낙관적일 수 있음
#
# 사용법: python scripts/16_tune_nested.py
import argparse, importlib, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate import (HIST, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics, best_threshold, clean,
                      feature_columns, load, split_fit_calib)

tune = importlib.import_module("15_tune")


def score_on(params, fit_all, cal_all, fit_fold, cal_fold, feats, fold_ids):
    ys, ss = [], []
    for j in fold_ids:
        rest = [f for f in fold_ids if f != j]
        m = tune.fit(params, fit_all[np.isin(fit_fold, rest)], cal_all[np.isin(cal_fold, rest)], feats)
        te = cal_all[cal_fold == j]
        ys.append(te[TARGET].to_numpy())
        ss.append(m.predict_proba(clean(te[feats]))[:, 1])
    return average_precision_score(np.concatenate(ys), np.concatenate(ss))


def main():
    p = argparse.ArgumentParser(description="중첩 교차검증으로 튜닝 검증")
    p.add_argument("--dst", default=RESULTS_DIR)
    args = p.parse_args()

    tr, va = load("Training"), load("Validation")
    folds = pd.read_csv(os.path.join(args.dst, "unit_split_assign.csv")).set_index("dev")["fold"]
    n_folds = int(folds.max()) + 1
    fit_all, cal_all = split_fit_calib(tr)
    fill = float(fit_all[HIST].max())
    fit_all, cal_all, va_b = (add_history_flag(x, fill) for x in (fit_all, cal_all, va))
    feats = feature_columns(fit_all)
    fit_fold, cal_fold = fit_all["dev"].map(folds).to_numpy(), cal_all["dev"].map(folds).to_numpy()
    va_fold = va["dev"].map(folds).to_numpy()

    # 후보는 15번이 고정 시드로 뽑은 21개. 15번의 점수는 쓰지 않음
    search = pd.read_csv(os.path.join(args.dst, "tune_search.csv")).sort_values("trial")
    cands = []
    for _, r in search.iterrows():
        c = {k: r[k] for k in tune.SPACE}
        c["num_leaves"], c["min_child_samples"] = int(c["num_leaves"]), int(c["min_child_samples"])
        cands.append(c)

    parts, fold_rows = [], []
    for k in range(n_folds):
        inner = [f for f in range(n_folds) if f != k]
        scores = [score_on(c, fit_all, cal_all, fit_fold, cal_fold, feats, inner) for c in cands]
        best = int(np.argmax(scores))
        keep_fit, keep_cal = fit_fold != k, cal_fold != k
        m = tune.fit(cands[best], fit_all[keep_fit], cal_all[keep_cal], feats)
        cal = cal_all[keep_cal]
        thr = best_threshold(cal[TARGET].to_numpy(), m.predict_proba(clean(cal[feats]))[:, 1])
        te = va_b[va_fold == k]
        s = m.predict_proba(clean(te[feats]))[:, 1]
        parts.append((s, s >= thr))
        fold_rows.append(dict(fold=k, chosen_trial=search["trial"].iloc[best], inner_pr_auc=scores[best],
                              pr_auc=average_precision_score(te[TARGET], s), **cands[best]))
        print(f"묶음 {k}: 후보 {search['trial'].iloc[best]} 선택, 내부 {scores[best]:.4f}, Validation {fold_rows[-1]['pr_auc']:.4f}",
              flush=True)

    va_sorted = pd.concat([va_b[va_fold == k] for k in range(n_folds)])
    s = np.concatenate([x[0] for x in parts])
    alarm = np.concatenate([x[1] for x in parts])
    res = pd.DataFrame([dict(setting="처음 보는 설비", model="LightGBM (중첩 튜닝)", positive_rate=va_sorted[TARGET].mean(),
                             pr_auc=average_precision_score(va_sorted[TARGET], s), **alarm_metrics(va_sorted, alarm))])
    res.to_csv(os.path.join(args.dst, "tune_nested_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.dst, "tune_nested_folds.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).round(4).to_string(index=False))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
