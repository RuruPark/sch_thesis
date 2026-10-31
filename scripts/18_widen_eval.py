# -*- coding: utf-8 -*-
# 평가 대상을 펌프에서 6종까지 넓혀, 설비종류마다 처음 보는 설비 성능을 따로 봄
# 입력: data/features_v2(또는 --src)  출력: data/results/widen{_태그}_metrics.csv, _folds.csv, _alarm.csv
# 설비종류마다 17번과 같은 방식으로 5묶음을 나누고, 묶음 설비는 Training까지 학습에서 빼서 "처음 보는 설비"로 둠
# 조기 종료와 임계값은 평가 설비종류의 학습 설비 뒤 20%로 정함. 펌프는 17번과 같은 조건이라 17번 결과가 재현돼야 함
#
# 사용법: python scripts/18_widen_eval.py --src data/features_v2 [--eval-types 6.가열로 ...] [--only-pooled] [--meta] [--tag 이름]
import argparse, glob, importlib, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve

from evaluate import (FACILITY, FEATURES_DIR, HIST, LGBM_PARAMS, RESULTS_DIR, TARGET, add_history_flag, alarm_metrics,
                      clean, feature_columns, split_fit_calib, sustained)

ex17 = importlib.import_module("17_extra_facilities")
EVAL_TYPES = [FACILITY] + ex17.EXTRA_TYPES
N_FOLDS = ex17.N_FOLDS
WAIT_MINUTES = [1, 2, 3, 5]


def load_split(src, split, t, target, neg_frac, rng):
    # 펌프 외 설비종류 Training은 17번과 같이 음성만 표본추출하고 가중치로 되돌림. 평가에 쓰는 Validation은 전부 읽음
    out = []
    for f in sorted(glob.glob(os.path.join(src, split, t, "*.parquet"))):
        d = pd.read_parquet(f)
        d = d[d[target].notna()]
        if neg_frac < 1:
            d = d[(d[target] == 1).to_numpy() | (rng.random(len(d)) < neg_frac)]
        num = d.select_dtypes("float64").columns
        d[num] = d[num].astype("float32")
        d["_w"] = np.where(d[target] == 1, 1.0, 1.0 / neg_frac).astype("float32")
        out.append(d)
    return pd.concat(out) if out else None


def weighted_threshold(y, score, w):
    # 표본추출한 음성을 가중치 없이 셀 경우, 정밀도가 부풀어 임계값이 낮게 잡히는 문제 있음
    p, r, t = precision_recall_curve(y, score, sample_weight=w)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return t[np.argmax(f1[:-1])]


def fit_lgbm(fit, cal, feats, target):
    return lgb.LGBMClassifier(**LGBM_PARAMS).fit(
        clean(fit[feats]), fit[target].to_numpy(), sample_weight=fit["_w"].to_numpy(),
        eval_set=[(clean(cal[feats]), cal[target].to_numpy())], eval_sample_weight=[cal["_w"].to_numpy()],
        eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    p = argparse.ArgumentParser(description="설비종류별 처음 보는 설비 평가")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--eval-types", nargs="*", default=EVAL_TYPES)
    p.add_argument("--neg-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--meta", action="store_true", help="설비 정보 특징 추가 (17번 --meta와 같음)")
    p.add_argument("--only-pooled", action="store_true", help="같은 종류만 학습한 모델을 건너뜀")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    target = args.target
    prefix = os.path.join(args.dst, "widen" + (f"_{args.tag}" if args.tag else ""))
    rng = np.random.default_rng(args.seed)

    tr = pd.concat([load_split(args.src, "Training", t, target, 1.0 if t == FACILITY else args.neg_frac, rng)
                    for t in EVAL_TYPES])
    types = list(tr["facility"].unique())
    if args.meta:
        meta = os.path.join(os.path.dirname(os.path.abspath(args.src)), "processed", "meta.csv")
        ex17.add_meta((tr,), meta)
    print(f"Training {tr['facility'].nunique()}종 {tr['dev'].nunique()}대 {len(tr):,}행(양성 {int(tr[target].sum()):,})")
    # 묶음 배정은 양성을 전부 남겨 표본추출과 상관없이 17번 펌프 배정과 같음. 설비 번호가 종류 사이에 겹칠 수 있어 종류마다 따로 둠
    folds = {t: ex17.assign_folds(tr[tr["facility"] == t], target, args.seed) for t in types}
    rank = tr.groupby(["facility", "dev"]).cumcount() / tr.groupby(["facility", "dev"])["dev"].transform("size")
    fit_mask = (rank < 0.8).to_numpy()  # split_fit_calib과 같은 기준

    metric_rows, fold_rows, alarm_rows = [], [], []
    for t in [t for t in args.eval_types if t in types]:
        va = load_split(args.src, "Validation", t, target, 1.0, rng)
        if args.meta:
            ex17.add_meta((va,), meta)
        va_fold = va["dev"].map(folds[t])
        pooled = {}
        for k in range(N_FOLDS):
            te_devs = [d for d, f in folds[t].items() if f == k]
            same = (tr["facility"] == t).to_numpy()
            keep = ~(same & tr["dev"].isin(te_devs).to_numpy())
            fit_all = tr[keep & fit_mask]
            cal = tr[keep & same & ~fit_mask]
            fill = float(fit_all[HIST].max())
            fit_all, cal = add_history_flag(fit_all, fill), add_history_flag(cal, fill)
            te = add_history_flag(va[va_fold == k], fill)
            if te.empty or cal[target].sum() == 0:
                print(f"{t} 묶음 {k}: 평가 설비 또는 보정 양성이 없어 건너뜀")
                continue
            feats = [c for c in feature_columns(fit_all) if c != "_w"]

            models = {"규칙": None}
            if not args.only_pooled:
                models["LightGBM 같은 종류만"] = fit_lgbm(fit_all[fit_all["facility"] == t], cal, feats, target)
            models["LightGBM 전체 종류"] = fit_lgbm(fit_all, cal, feats, target)
            y, w_cal = te[target].to_numpy(), cal["_w"].to_numpy()
            for name, m in models.items():
                if m is None:
                    s_te, s_cal = te["전류고조파평균"].fillna(-1).to_numpy(), cal["전류고조파평균"].fillna(-1).to_numpy()
                else:
                    s_te, s_cal = m.predict_proba(clean(te[feats]))[:, 1], m.predict_proba(clean(cal[feats]))[:, 1]
                thr = weighted_threshold(cal[target].to_numpy(), s_cal, w_cal)
                pooled.setdefault(name, []).append((k, s_te, s_te >= thr))
                fold_rows.append(dict(facility=t, fold=k, model=name, units=te["dev"].nunique(), positives=int(y.sum()),
                                      pr_auc=average_precision_score(y, s_te) if y.sum() else np.nan))
            print(f"{t} 묶음 {k}: " + ", ".join(f"{r['model']} {r['pr_auc']:.3f}" for r in fold_rows[-len(models):]))

        for name, parts in pooled.items():
            te_all = pd.concat([va[va_fold == k] for k, _, _ in parts])
            y_all = te_all[target].to_numpy()
            s = np.concatenate([x[1] for x in parts])
            alarm = np.concatenate([x[2] for x in parts])
            metric_rows.append(dict(facility=t, model=name, units=te_all["dev"].nunique(),
                                    units_with_pos=int(te_all.groupby("dev")[target].max().sum()),
                                    src=os.path.basename(args.src.rstrip("/\\")), meta=args.meta, seed=args.seed,
                                    pr_auc=average_precision_score(y_all, s), **alarm_metrics(te_all, alarm, target)))
            for n in WAIT_MINUTES:
                alarm_rows.append(dict(facility=t, model=name, wait_min=n,
                                       **alarm_metrics(te_all, sustained(te_all, alarm, n), target)))
        # 설비종류 하나 끝날 때마다 저장. 중간에 멈춰도 끝난 종류 결과는 남김
        pd.DataFrame(metric_rows).to_csv(prefix + "_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(prefix + "_folds.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(alarm_rows).to_csv(prefix + "_alarm.csv", index=False, encoding="utf-8-sig")
        del va

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        res = pd.DataFrame(metric_rows)
        print(res[["facility", "model", "units", "units_with_pos", "pr_auc", "f1", "event_recall",
                   "false_alarm_per_day"]].round(3).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index=["facility", "model"], columns="fold", values="pr_auc").round(3))
    print("완료:", prefix)


if __name__ == "__main__":
    main()
