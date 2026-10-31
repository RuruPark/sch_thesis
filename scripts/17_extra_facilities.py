# -*- coding: utf-8 -*-
# 다른 설비종류를 학습에 더하면 처음 보는 펌프/일반모터 설비 성능이 오르는지
# 입력: data/features  출력: data/results/extra_fac{_태그}_metrics.csv, _folds.csv
# 평가 틀은 08번과 같음(펌프 설비 5묶음, 묶음 설비의 Validation 기간). 학습 데이터만 "펌프" 대 "펌프 + 다른 설비종류 Training"
# 조기 종료와 임계값은 두 모델 모두 펌프 학습 설비의 뒤 20%로 고정해, 차이가 학습 데이터에서만 나오게 함
#
# 사용법: python scripts/17_extra_facilities.py [--types 3.공기압축기 ...] [--neg-frac 0.2] [--seed 1] [--type-feature] [--only-extra]
#         [--compare] [--save-scores] [--tag 이름]
import argparse, glob, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (FEATURES_DIR, HIST, LGBM_PARAMS, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics,
                      best_threshold, clean, feature_columns, load, split_fit_calib, sustained)

N_FOLDS = 5
WAIT_MINUTES = [1, 2, 3, 5, 10]  # 09번과 같은 대기 시간
# 02b_eda_facility.py에서 펌프 기준 전류고조파 경계(30/60)로 라벨이 97~99% 재현된 설비종류
# 보일러(93%), 사출기(93%, 경고 62행)는 경계가 덜 맞고 이상 발생도 100건 남짓이라 뺌
EXTRA_TYPES = ["3.공기압축기", "4.공작기계", "6.가열로", "7.압출기", "9.주조기", "10.열처리"]


META_COLS = {"facility_capacity": "정격용량", "facility_volt": "정격전압", "facility_year": "제작연도",
             "CONTRACT_POWER": "계약전력"}


def add_meta(ds, meta_path):
    # 처음 보는 설비에서 "어떤 급의 설비인지"를 알려 주려는 값. 같은 전류라도 설비 크기에 따라 의미가 달라서 정격 대비 부하도 같이 줌
    m = pd.read_csv(meta_path).drop_duplicates("dev").set_index("dev")[list(META_COLS)].rename(columns=META_COLS)
    m = m.astype("float32")
    for d in ds:
        for c in m.columns:
            d[c] = d["dev"].map(m[c]).astype("float32")
        d["부하_정격비"] = (d["유효전력평균"] / d["정격용량"].where(d["정격용량"] > 0)).astype("float32")


def assign_folds(tr, target, seed=0):
    # seed 0은 08번과 같은 배정. 다른 seed는 양성 수 순서 5대씩 묶음 안에서만 묶음 번호를 섞어, 묶음마다 양성 수가 고르게 남게 함
    pos = tr.groupby("dev")[target].sum().sort_values(ascending=False, kind="stable")
    rng = np.random.default_rng(seed)
    folds = {}
    for devs in (pos[pos > 0].index, pos[pos == 0].index):
        for start in range(0, len(devs), N_FOLDS):
            block = devs[start:start + N_FOLDS]
            order = rng.permutation(N_FOLDS)[:len(block)] if seed else np.arange(len(block))
            folds.update({d: int(k) for d, k in zip(block, order)})
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


def dev_balance(d, target, w):
    # 사건의 절반 가까이가 상위 5대에 몰려 그 설비의 버릇을 배우기 쉬움 → 설비마다 양성 가중치 합을 평균에 맞춤. 극단값은 0.2~5배로 자름
    pos = (d[target] == 1).to_numpy()
    dev = d["dev"].to_numpy()
    devs, inv, cnt = np.unique(dev[pos], return_inverse=True, return_counts=True)
    out = np.asarray(w, dtype="float64").copy()
    out[pos] *= np.clip(cnt.mean() / cnt[inv], 0.2, 5)
    return out


def fit_lgbm(fit, cal, feats, target, weight=None):
    return lgb.LGBMClassifier(**LGBM_PARAMS).fit(
        clean(fit[feats]), fit[target].to_numpy(), sample_weight=weight,
        eval_set=[(clean(cal[feats]), cal[target].to_numpy())], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)])


COMPARE_MAX_ROWS = 1_500_000


def compare_sample(d, target, w, seed=0):
    # LR·RF는 표준화·대치 단계에서 float64 사본을 만들어, 펌프+6종 전체(수백만 행 × 167)를 넣을 경우 16GB에서 메모리가 모자라는 문제 있음
    # 양성은 전부 두고 음성만 더 줄인 뒤 가중치로 되돌림
    neg = (d[target] == 0).to_numpy()
    frac = min(1.0, max(COMPARE_MAX_ROWS - int((~neg).sum()), 1) / max(int(neg.sum()), 1))
    keep = ~neg | (np.random.default_rng(seed).random(len(d)) < frac)
    return d[keep], np.where(neg[keep], w[keep] / frac, w[keep])


def fit_compare(fit, feats, target, w):
    # 06·07번과 같은 설정. 펌프+추가 데이터로 학습해 LightGBM과 같은 조건에서 비교
    sub, sw = compare_sample(fit, target, w)
    X, y = clean(sub[feats]), sub[target].to_numpy()
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000))
    lr.fit(X, y, logisticregression__sample_weight=sw)
    # 트리마다 학습 행의 10%만 씀. 전체 행으로 키울 경우, 4코어에서 한 시간 넘게 걸리는 문제 있음
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=200, max_samples=0.1, min_samples_leaf=20,
                                              class_weight="balanced_subsample", n_jobs=-1, random_state=0))
    rf.fit(X, y, randomforestclassifier__sample_weight=sw)
    return {"Logistic Regression 펌프+추가": lr, "Random Forest 펌프+추가": rf}


def main():
    p = argparse.ArgumentParser(description="다른 설비종류를 학습에 더한 처음 보는 펌프 설비 평가")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--types", nargs="*", default=EXTRA_TYPES)
    p.add_argument("--neg-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="묶음 배정 시드. 0은 08번과 같은 배정")
    p.add_argument("--type-feature", action="store_true", help="설비종류 번호를 특징으로 추가")
    p.add_argument("--meta", action="store_true", help="설비 정보(정격 용량·전압, 제작 연도, 계약전력) 특징 추가")
    p.add_argument("--dev-balance", action="store_true", help="설비마다 양성 가중치 합을 맞춤")
    p.add_argument("--only-extra", action="store_true", help="펌프만 모델을 건너뜀 (민감도 확인용)")
    p.add_argument("--compare", action="store_true", help="Logistic Regression, Random Forest도 펌프+추가 데이터로 학습해 비교")
    p.add_argument("--save-scores", action="store_true", help="처음 보는 설비 행 단위 점수와 경보를 저장 (신뢰구간, 임계값 재보정용)")
    p.add_argument("--shap", action="store_true", help="묶음마다 처음 보는 설비에서 펌프+추가 모델의 SHAP 기여도를 모음")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    target = args.target
    prefix = os.path.join(args.dst, "extra_fac" + (f"_{args.tag}" if args.tag else ""))

    tr, va = load("Training", args.src, target), load("Validation", args.src, target)
    ex = load_extra(args.src, args.types, target, args.neg_frac)
    print(f"펌프 Training {len(tr):,}행, Validation {len(va):,}행 / 추가 {ex['facility'].nunique()}종 "
          f"{ex['dev'].nunique()}대 {len(ex):,}행(양성 {int(ex[target].sum()):,})")
    if args.meta:
        add_meta((tr, va, ex), os.path.join(os.path.dirname(os.path.abspath(args.src)), "processed", "meta.csv"))
    if args.type_feature:
        for d in (tr, va, ex):
            d["설비종류"] = d["facility"].str.split(".").str[0].astype("float32")
    folds = assign_folds(tr, target, args.seed)
    tr_fold, va_fold = tr["dev"].map(folds), va["dev"].map(folds)

    pooled, fold_rows, shap_parts, thr_rows = {}, [], [], []
    for k in range(N_FOLDS):
        fit_p, cal_p = split_fit_calib(tr[tr_fold != k])
        fit_x, _ = split_fit_calib(ex)
        fill = float(pd.concat([fit_p[HIST], fit_x[HIST]]).max())
        fit_p, cal_p, fit_x = (add_history_flag(d, fill) for d in (fit_p, cal_p, fit_x))
        te = add_history_flag(va[va_fold == k], fill)
        feats = feature_columns(fit_p)
        feats = [c for c in feats if c != "_w"]

        both = pd.concat([fit_p.assign(_w=1.0), fit_x])
        models = {"규칙": None}
        if not args.only_extra:
            w_p = dev_balance(fit_p, target, np.ones(len(fit_p))) if args.dev_balance else None
            models["LightGBM 펌프만"] = fit_lgbm(fit_p, cal_p, feats, target, w_p)
        w_b = both["_w"].to_numpy()
        if args.dev_balance:
            w_b = dev_balance(both, target, w_b)
        models["LightGBM 펌프+추가"] = fit_lgbm(both, cal_p, feats, target, w_b)
        if args.compare:
            models.update(fit_compare(both, feats, target, both["_w"].to_numpy()))
        y = te[target].to_numpy()
        for name, m in models.items():
            if m is None:
                s_te, s_cal = te["전류고조파평균"].fillna(-1).to_numpy(), cal_p["전류고조파평균"].fillna(-1).to_numpy()
            else:
                s_te, s_cal = m.predict_proba(clean(te[feats]))[:, 1], m.predict_proba(clean(cal_p[feats]))[:, 1]
            thr = best_threshold(cal_p[target].to_numpy(), s_cal)
            pooled.setdefault(name, []).append((s_te, s_te >= thr))
            thr_rows.append(dict(fold=k, model=name, threshold=float(thr)))
            fold_rows.append(dict(fold=k, model=name, units=te["dev"].nunique(), positives=int(y.sum()),
                                  pr_auc=average_precision_score(y, s_te)))
        if args.shap:
            # 처음 보는 설비 기준 해석. 마지막 열은 기준값(bias), 값은 log-odds 단위
            c = models["LightGBM 펌프+추가"].predict_proba(clean(te[feats]), pred_contrib=True)[:, :-1]
            shap_parts.append((feats, c, y))
        print(f"묶음 {k}: " + ", ".join(f"{r['model']} {r['pr_auc']:.3f}" for r in fold_rows[-len(models):]))
        del both

    va_sorted = pd.concat([va[va_fold == k] for k in range(N_FOLDS)])
    y_all = va_sorted[target].to_numpy()
    rows = []
    for name, parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        rows.append(dict(model=name, types=" ".join(args.types), neg_frac=args.neg_frac, seed=args.seed,
                         type_feature=args.type_feature, dev_balance=args.dev_balance, meta=args.meta, src=os.path.basename(args.src.rstrip("/\\")),
                         pr_auc=average_precision_score(y_all, s), **alarm_metrics(va_sorted, alarm, target)))
    res = pd.DataFrame(rows)
    res.to_csv(prefix + "_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(prefix + "_folds.csv", index=False, encoding="utf-8-sig")
    alarm_rows = []
    for name, parts in pooled.items():
        raw = np.concatenate([x[1] for x in parts])
        for n in WAIT_MINUTES:
            alarm_rows.append(dict(model=name, wait_min=n, **alarm_metrics(va_sorted, sustained(va_sorted, raw, n), target)))
    pd.DataFrame(alarm_rows).to_csv(prefix + "_alarm.csv", index=False, encoding="utf-8-sig")

    if args.save_scores:
        # 행 순서는 va_sorted와 같음. 임계값은 묶음마다 펌프 학습 설비 뒤 20%에서 고른 값
        sc = pd.DataFrame({"time": va_sorted.index, "dev": va_sorted["dev"].to_numpy(),
                           "fold": va_sorted["dev"].map(folds).to_numpy(), target: y_all,
                           "minutes_to_event": va_sorted["minutes_to_event"].to_numpy()})
        for i, (name, parts) in enumerate(pooled.items()):
            sc[f"score_{i}"] = np.concatenate([x[0] for x in parts]).astype("float32")
        sc.to_parquet(prefix + "_scores.parquet", index=False)
        names = pd.DataFrame({"column": [f"score_{i}" for i in range(len(pooled))], "model": list(pooled)})
        thr = pd.DataFrame(thr_rows).merge(names, on="model")
        thr.to_csv(prefix + "_thresholds.csv", index=False, encoding="utf-8-sig")

    if shap_parts:
        feats = shap_parts[0][0]
        c = np.concatenate([x[1] for x in shap_parts])
        y = np.concatenate([x[2] for x in shap_parts])
        imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(c).mean(axis=0),
                            "mean_shap_pos": c[y == 1].mean(axis=0), "mean_shap_neg": c[y == 0].mean(axis=0)})
        imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(prefix + "_shap.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(alarm_rows).round(3).to_string(index=False))
        print(res.drop(columns="types").round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index="model", columns="fold", values="pr_auc").round(3))
    print("완료:", prefix)


if __name__ == "__main__":
    main()
