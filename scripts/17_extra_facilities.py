# -*- coding: utf-8 -*-
# 다른 설비종류를 학습에 더하면 처음 보는 펌프/일반모터 설비 성능이 오르는지
# 입력: data/features  출력: data/results/extra_fac{_태그}_metrics.csv, _folds.csv
# 평가 틀은 08번과 같음(펌프 설비 5묶음, 묶음 설비의 Validation 기간). 학습 데이터만 "펌프" 대 "펌프 + 다른 설비종류 Training"
# 조기 종료와 임계값은 두 모델 모두 펌프 학습 설비의 뒤 20%로 고정해, 차이가 학습 데이터에서만 나오게 함
#
# 사용법: python scripts/17_extra_facilities.py [--types 3.공기압축기 ...] [--neg-frac 0.2] [--tag 이름]
import argparse, glob, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from evaluate import (FEATURES_DIR, HIST, LGBM_PARAMS, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics,
                      best_threshold, clean, feature_columns, load, split_fit_calib)

N_FOLDS = 5
# 02b_eda_facility.py에서 펌프 기준 전류고조파 경계(30/60)로 라벨이 97~99% 재현된 설비종류
# 보일러(93%), 사출기(93%, 경고 62행)는 경계가 덜 맞고 이상 발생도 100건 남짓이라 뺌
EXTRA_TYPES = ["3.공기압축기", "4.공작기계", "6.가열로", "7.압출기", "9.주조기", "10.열처리"]


def assign_folds(tr, target):
    # 08번과 같은 배정. 펌프만으로 정해야 기준 결과(0.470)와 같은 묶음이 됨
    pos = tr.groupby("dev")[target].sum().sort_values(ascending=False, kind="stable")
    with_pos, without = pos[pos > 0].index, pos[pos == 0].index
    folds = {d: i % N_FOLDS for i, d in enumerate(with_pos)}
    folds.update({d: i % N_FOLDS for i, d in enumerate(without)})
    return folds


def load_extra(src, types, target, neg_frac, seed=0):
    # 6종 Training을 다 올릴 경우 1,000만 행이 넘어 16GB에서 메모리가 모자람 → 파일마다 음성만 표본추출하고 가중치로 되돌림
    rng = np.random.default_rng(seed)
    out = []
    for t in types:
        for f in sorted(glob.glob(os.path.join(src, "Training", t, "*.parquet"))):
            d = pd.read_parquet(f)
            d = d[d[target].notna()]
            keep = (d[target] == 1).to_numpy() | (rng.random(len(d)) < neg_frac)
            d = d[keep]
            num = d.select_dtypes("float64").columns
            d[num] = d[num].astype("float32")
            d["_w"] = np.where(d[target] == 1, 1.0, 1.0 / neg_frac).astype("float32")
            out.append(d)
    return pd.concat(out)


def fit_lgbm(fit, cal, feats, target, weight=None):
    return lgb.LGBMClassifier(**LGBM_PARAMS).fit(
        clean(fit[feats]), fit[target].to_numpy(), sample_weight=weight,
        eval_set=[(clean(cal[feats]), cal[target].to_numpy())], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    p = argparse.ArgumentParser(description="다른 설비종류를 학습에 더한 처음 보는 펌프 설비 평가")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--types", nargs="*", default=EXTRA_TYPES)
    p.add_argument("--neg-frac", type=float, default=0.2)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    target = args.target
    prefix = os.path.join(args.dst, "extra_fac" + (f"_{args.tag}" if args.tag else ""))

    tr, va = load("Training", args.src, target), load("Validation", args.src, target)
    ex = load_extra(args.src, args.types, target, args.neg_frac)
    print(f"펌프 Training {len(tr):,}행, Validation {len(va):,}행 / 추가 {ex['facility'].nunique()}종 "
          f"{ex['dev'].nunique()}대 {len(ex):,}행(양성 {int(ex[target].sum()):,})")
    folds = assign_folds(tr, target)
    tr_fold, va_fold = tr["dev"].map(folds), va["dev"].map(folds)

    pooled, fold_rows = {}, []
    for k in range(N_FOLDS):
        fit_p, cal_p = split_fit_calib(tr[tr_fold != k])
        fit_x, _ = split_fit_calib(ex)
        fill = float(pd.concat([fit_p[HIST], fit_x[HIST]]).max())
        fit_p, cal_p, fit_x = (add_history_flag(d, fill) for d in (fit_p, cal_p, fit_x))
        te = add_history_flag(va[va_fold == k], fill)
        feats = feature_columns(fit_p)
        feats = [c for c in feats if c != "_w"]

        both = pd.concat([fit_p.assign(_w=1.0), fit_x])
        models = {
            "규칙": None,
            "LightGBM 펌프만": fit_lgbm(fit_p, cal_p, feats, target),
            "LightGBM 펌프+추가": fit_lgbm(both, cal_p, feats, target, both["_w"].to_numpy()),
        }
        y = te[target].to_numpy()
        for name, m in models.items():
            if m is None:
                s_te, s_cal = te["전류고조파평균"].fillna(-1).to_numpy(), cal_p["전류고조파평균"].fillna(-1).to_numpy()
            else:
                s_te, s_cal = m.predict_proba(clean(te[feats]))[:, 1], m.predict_proba(clean(cal_p[feats]))[:, 1]
            thr = best_threshold(cal_p[target].to_numpy(), s_cal)
            pooled.setdefault(name, []).append((s_te, s_te >= thr))
            fold_rows.append(dict(fold=k, model=name, units=te["dev"].nunique(), positives=int(y.sum()),
                                  pr_auc=average_precision_score(y, s_te)))
        print(f"묶음 {k}: " + ", ".join(f"{r['model']} {r['pr_auc']:.3f}" for r in fold_rows[-3:]))
        del both

    va_sorted = pd.concat([va[va_fold == k] for k in range(N_FOLDS)])
    y_all = va_sorted[target].to_numpy()
    rows = []
    for name, parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        rows.append(dict(model=name, types=" ".join(args.types), neg_frac=args.neg_frac,
                         pr_auc=average_precision_score(y_all, s), **alarm_metrics(va_sorted, alarm, target)))
    res = pd.DataFrame(rows)
    res.to_csv(prefix + "_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(prefix + "_folds.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.drop(columns="types").round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index="model", columns="fold", values="pr_auc").round(3))
    print("완료:", prefix)


if __name__ == "__main__":
    main()
