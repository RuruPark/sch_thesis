# -*- coding: utf-8 -*-
# 30분 내 이상 발생 예측 베이스라인 비교: 항상 음성, 전류고조파 임계값 규칙, Logistic Regression
# 입력: data/features  출력: data/results/baseline_metrics.csv, baseline_by_facility.csv
# Training으로 학습과 경보 임계값 선택, Validation(같은 설비의 뒤 기간)으로 평가
#
# 사용법: python scripts/06_baseline.py
import argparse, glob, os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "target_30"
NON_FEATURES = ["dev", "facility", "target_10", "target_30", "target_60", "minutes_to_event"]


def load(split, src):
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(src, split, "*", "*.parquet")))]
    d = pd.concat(parts)
    return d[d[TARGET].notna()].copy()


def best_threshold(y, score):
    # Validation을 보고 임계값을 고를 경우, 성능이 부풀려지는 문제 있어 Training에서 F1 최대 지점으로 고정
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return t[np.argmax(f1[:-1])]


def alarm_metrics(d, alarm):
    y = d[TARGET].to_numpy()
    tp = int((alarm & (y == 1)).sum())
    precision = tp / max(alarm.sum(), 1)
    recall = tp / max((y == 1).sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # 사건 단위: 이상 발생 30분 전 안에서 한 번이라도 경보가 울렸는지. 리드 타임은 가장 이른 경보 기준
    pos = d[y == 1].assign(alarm=alarm[y == 1])
    event_time = pos.index + pd.to_timedelta(pos["minutes_to_event"], unit="min")
    ev = pos.groupby([pos["dev"].to_numpy(), event_time]).apply(
        lambda g: g.loc[g["alarm"], "minutes_to_event"].max())
    caught = ev.notna()

    # 오경보: 이어진 경보 구간 중 30분 안에 이상 발생이 없었던 구간 수. 하루 단위는 설비 1대가 정상 운전한 하루 기준
    t = d.index.to_series().reset_index(drop=True)
    dev = d["dev"].reset_index(drop=True)
    new_run = (dev != dev.shift()) | (t - t.shift() != pd.Timedelta(minutes=1)) | ~pd.Series(alarm).shift(fill_value=False)
    run_id = (new_run & alarm).cumsum().where(alarm)
    run_has_pos = pd.Series(y == 1).groupby(run_id).any()
    false_runs = int((~run_has_pos).sum())
    unit_days = len(d) / 1440

    return dict(precision=precision, recall=recall, f1=f1,
                events=len(ev), event_recall=caught.mean(),
                lead_median=ev[caught].median(), false_alarm_per_day=false_runs / unit_days)


def main():
    p = argparse.ArgumentParser(description="베이스라인 비교")
    p.add_argument("--src", default=os.path.join(PROJECT_ROOT, "data", "features"))
    p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "data", "results"))
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    tr, va = load("Training", args.src), load("Validation", args.src)
    feats = [c for c in tr.columns if c not in NON_FEATURES]
    y_tr, y_va = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    print(f"Training {len(tr):,}행 양성 {y_tr.mean():.4f} / Validation {len(va):,}행 양성 {y_va.mean():.4f}")

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000))
    X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
    lr.fit(X_tr, y_tr)

    # 규칙 점수가 비어 있으면 경보를 낼 근거가 없으므로 가장 낮은 값으로 둠
    scores = {
        "항상 음성": (np.zeros(len(tr)), np.zeros(len(va))),
        "규칙: 현재 전류고조파": (tr["전류고조파평균"].fillna(-1).to_numpy(), va["전류고조파평균"].fillna(-1).to_numpy()),
        "규칙: 최근 10분 전류고조파 최댓값": (tr["전류고조파평균_max10"].fillna(-1).to_numpy(), va["전류고조파평균_max10"].fillna(-1).to_numpy()),
        "Logistic Regression": (lr.predict_proba(X_tr)[:, 1],
                                lr.predict_proba(va[feats].replace([np.inf, -np.inf], np.nan))[:, 1]),
    }

    rows, by_fac = [], []
    for name, (s_tr, s_va) in scores.items():
        if name == "항상 음성":
            thr, alarm = np.nan, np.zeros(len(va), dtype=bool)
        else:
            thr = best_threshold(y_tr, s_tr)
            alarm = s_va >= thr
        r = dict(model=name, threshold=thr, pr_auc=average_precision_score(y_va, s_va), **alarm_metrics(va, alarm))
        rows.append(r)
        for fac, idx in va.groupby("facility").indices.items():
            by_fac.append(dict(model=name, facility=fac, n=len(idx), positives=int(y_va[idx].sum()),
                               pr_auc=average_precision_score(y_va[idx], s_va[idx]) if y_va[idx].sum() else np.nan))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dst, "baseline_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(by_fac).to_csv(os.path.join(args.dst, "baseline_by_facility.csv"), index=False, encoding="utf-8-sig")

    coef = pd.Series(lr[-1].coef_[0], index=feats)
    coef.reindex(coef.abs().sort_values(ascending=False).index).to_csv(
        os.path.join(args.dst, "baseline_lr_coef.csv"), header=["coef"], encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
