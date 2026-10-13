# -*- coding: utf-8 -*-
# LightGBM이 어떤 특징을 보고 30분 내 이상 발생을 예측했는지 SHAP 값으로 확인
# 입력: data/features  출력: data/results/shap_importance.csv, shap_dependence_sample.parquet (그래프는 10번)
# shap 패키지 대신 LightGBM 내장 pred_contrib 사용. 트리 모델에서는 같은 TreeSHAP 값이 나옴
#
# 사용법: python scripts/12_shap.py [--norm]
import argparse, os
import numpy as np
import pandas as pd
import lightgbm as lgb

from evaluate import (FEATURES_DIR, HIST, LGBM_PARAMS, RESULTS_DIR, TARGET, add_history_flag, add_unit_norm, clean,
                      feature_columns, load, split_fit_calib)

TOP_N = 15
# 의존도 그래프용. Validation 전체를 찍을 경우, 점이 겹쳐 양성이 안 보이는 문제 있어 음성만 줄임
SAMPLE_NEG = 20000


def main():
    p = argparse.ArgumentParser(description="SHAP 해석")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--norm", action="store_true", help="설비별 평소 값 대비 특징 추가")
    args = p.parse_args()

    tr, va = load("Training", args.src), load("Validation", args.src)
    if args.norm:
        tr, va = add_unit_norm(tr, va)
    fit, cal = split_fit_calib(tr)
    fill = float(fit[HIST].max())
    fit, cal, va = (add_history_flag(x, fill) for x in (fit, cal, va))
    feats = feature_columns(fit)

    gbm = lgb.LGBMClassifier(**LGBM_PARAMS).fit(
        clean(fit[feats]), fit[TARGET], eval_set=[(clean(cal[feats]), cal[TARGET])], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"LightGBM 트리 {gbm.best_iteration_}개, 특징 {len(feats)}개")

    # 마지막 열은 기준값(bias). 값은 log-odds 단위
    contrib = gbm.predict_proba(clean(va[feats]), pred_contrib=True)[:, :-1]
    y = va[TARGET].to_numpy()
    imp = pd.DataFrame({"feature": feats,
                        "mean_abs_shap": np.abs(contrib).mean(axis=0),
                        "mean_shap_pos": contrib[y == 1].mean(axis=0),
                        "mean_shap_neg": contrib[y == 0].mean(axis=0)})
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(args.dst, "shap_importance.csv"), index=False, encoding="utf-8-sig")

    top = imp["feature"].iloc[0]
    rng = np.random.default_rng(0)
    keep = (y == 1) | (rng.random(len(y)) < SAMPLE_NEG / max((y == 0).sum(), 1))
    j = feats.index(top)
    pd.DataFrame({"feature": top, "value": va[top].to_numpy()[keep], "shap": contrib[keep, j],
                  "target": y[keep]}).to_parquet(os.path.join(args.dst, "shap_dependence_sample.parquet"))

    with pd.option_context("display.width", 200):
        print(imp.head(TOP_N).round(4).to_string(index=False))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
