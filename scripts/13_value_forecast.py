# -*- coding: utf-8 -*-
# 선행 연구(이제혁, 2023) 방식 재현: 30분 뒤 전류고조파 값을 먼저 예측하고, 예측값이 기준을 넘으면 경보
# 입력: data/features, data/cleaned, data/target, data/results/unit_split_assign.csv (08번 묶음)
# 출력: data/results/value_forecast_metrics.csv
# 선행 연구의 Transformer 대신 LightGBM 회귀를 써서 "값 예측 후 판정"과 "전환 직접 분류"의 방식 차이만 비교
#
# 사용법: python scripts/13_value_forecast.py
import argparse, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

from evaluate import (FEATURES_DIR, FACILITY, HIST, LGBM_PARAMS, PROJECT_ROOT, RESULTS_DIR, TARGET, add_history_flag,
                      alarm_metrics, best_threshold, clean, feature_columns, load, split_fit_calib)

HORIZON = 30
VALUE = "전류고조파평균"
LIMIT = 30.0  # 전류고조파 SOH '주의' 경계


def future_max(split, clean_dir, target_dir):
    # 다음 1~30분 운전 구간의 전류고조파 최댓값. 정지 구간은 고조파 비율이 부풀려져 빼고 계산
    out = []
    for f in sorted(os.listdir(os.path.join(clean_dir, split, FACILITY))):
        d = pd.read_parquet(os.path.join(clean_dir, split, FACILITY, f), columns=[VALUE])
        run = pd.read_parquet(os.path.join(target_dir, split, FACILITY, f), columns=["is_running"])["is_running"]
        s = d[VALUE].where(run == 1)
        fut = s[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1)
        out.append(pd.DataFrame({"dev": int(f[:-8]), "future_max": fut}))
    return pd.concat(out).set_index("dev", append=True)


def attach(d, fut):
    key = pd.MultiIndex.from_arrays([d.index, d["dev"].to_numpy()])
    return d.assign(future_max=fut["future_max"].reindex(key).to_numpy())


def train(tr):
    fit, cal_raw = split_fit_calib(tr)
    fill = float(fit[HIST].max())
    fit, cal = add_history_flag(fit, fill), add_history_flag(cal_raw, fill)
    feats = feature_columns(fit)
    feats = [c for c in feats if c != "future_max"]
    fit, cal_fit = fit[fit["future_max"].notna()], cal[cal["future_max"].notna()]
    params = {k: v for k, v in LGBM_PARAMS.items()}
    reg = lgb.LGBMRegressor(**params).fit(clean(fit[feats]), fit["future_max"],
                                         eval_set=[(clean(cal_fit[feats]), cal_fit["future_max"])],
                                         callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = lambda x: reg.predict(clean(add_history_flag(x, fill)[feats]))
    # 경보 기준 두 가지: 선행 연구처럼 기준값 30 그대로, 또는 분류 모델과 같이 뒤 20% 기간 F1 최대 지점
    return pred, {"기준값 30": LIMIT, "F1 최대 임계값": best_threshold(cal_raw[TARGET].to_numpy(), pred(cal_raw))}


def main():
    p = argparse.ArgumentParser(description="값 예측 후 판정 방식 재현")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--clean", default=os.path.join(PROJECT_ROOT, "data", "cleaned"))
    p.add_argument("--target-dir", default=os.path.join(PROJECT_ROOT, "data", "target"))
    p.add_argument("--dst", default=RESULTS_DIR)
    args = p.parse_args()

    tr = attach(load("Training", args.src), future_max("Training", args.clean, args.target_dir))
    va = attach(load("Validation", args.src), future_max("Validation", args.clean, args.target_dir))
    folds = pd.read_csv(os.path.join(args.dst, "unit_split_assign.csv")).set_index("dev")["fold"]
    tr_fold, va_fold = tr["dev"].map(folds), va["dev"].map(folds)
    n_folds = int(folds.max()) + 1
    print(f"Training {len(tr):,}행, Validation {len(va):,}행, 30분 뒤 최댓값 있는 행 {tr['future_max'].notna().mean():.1%}")

    seen = train(tr)
    pooled = {}
    for k in range(n_folds):
        te = va[va_fold == k]
        unseen = train(tr[tr_fold != k])
        for setting, (pred, thrs) in {"본 적 있는 설비": seen, "처음 보는 설비": unseen}.items():
            s = pred(te)
            for rule, thr in thrs.items():
                pooled.setdefault((setting, rule), []).append((s, s >= thr))
        print(f"묶음 {k} 완료")

    va_sorted = pd.concat([va[va_fold == k] for k in range(n_folds)])
    y = va_sorted[TARGET].to_numpy()
    rows = []
    for (setting, rule), parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        rows.append(dict(setting=setting, model=f"값 예측 후 판정 ({rule})", pr_auc=average_precision_score(y, s),
                         **alarm_metrics(va_sorted, alarm)))
    # 값 예측이 완벽해도 얼마나 맞히는지 확인용. 실제 다음 30분 최댓값을 점수로, 기준값 30으로 경보
    # 5분 미만 깜빡임도 30을 넘어, 음성 시점의 약 6%가 이 점수로는 양성처럼 보이는 문제 있음
    fut = va_sorted["future_max"].fillna(-1).to_numpy()
    rows.append(dict(setting="참고", model="완벽한 값 예측 (실제 다음 30분 최댓값)", pr_auc=average_precision_score(y, fut),
                     **alarm_metrics(va_sorted, fut >= LIMIT)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dst, "value_forecast_metrics.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
