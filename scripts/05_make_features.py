# -*- coding: utf-8 -*-
# 조기 경보 모델 입력 특징: 최근 10/30/60분 통계, 상간 불균형, 최근 이상 이력, 시간대
# 입력: data/cleaned, data/target  출력: data/features/<split>/<설비종류>/<번호>.parquet (예측 대상 시점만)
# 모든 특징은 그 시점까지의 값만 씀. 미래 값을 섞을 경우, 검증 성능이 실제보다 높게 나오는 문제 있음
#
# 사용법: python scripts/05_make_features.py
import argparse, glob, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["전류평균", "전류고조파평균", "전압고조파평균", "유효전력평균", "무효전력평균",
        "역률평균", "상전압평균", "온도", "주파수"]
WINDOWS = [10, 30, 60]
TARGETS = ["target_10", "target_30", "target_60", "minutes_to_event"]


def imbalance(d, item):
    x = d[[f"{p}상{item}" for p in "RST"]]
    mean = x.mean(axis=1)
    # 평균이 0에 가까울 경우, 비율이 무한대로 튀는 문제 있어 1 미만은 비움
    return ((x.max(axis=1) - x.min(axis=1)) / mean).where(mean >= 1)


def make_features(d, t):
    s = d[BASE].astype("float32").copy()
    s["전류불균형"] = imbalance(d, "전류")
    s["전류고조파불균형"] = imbalance(d, "전류고조파")
    feats = {c: s[c] for c in s.columns}
    for w in WINDOWS:
        # 창의 절반 넘게 비어 있으면 통계가 대표성이 없어서 비움
        r = s.rolling(w, min_periods=w // 2)
        for c in s.columns:
            feats[f"{c}_mean{w}"] = r[c].mean()
            feats[f"{c}_std{w}"] = r[c].std()
            feats[f"{c}_max{w}"] = r[c].max()
            feats[f"{c}_diff{w}"] = s[c] - s[c].shift(w)
    f = pd.DataFrame(feats, index=d.index)

    abn = t["is_abnormal"].astype("float32")
    f["이상비율_60"] = abn.rolling(60, min_periods=30).mean()
    idx = pd.Series(np.arange(len(t), dtype="float64"), index=t.index)
    last_abn = idx.where(abn == 1).ffill()
    f["마지막이상후_분"] = idx - last_abn
    run = t["is_running"].astype(bool)
    run_start = idx.where(run & ~run.shift(fill_value=False)).ffill()
    f["운전지속_분"] = (idx - run_start).where(run)
    f["시"] = d.index.hour.astype("int8")
    f["요일"] = d.index.dayofweek.astype("int8")
    f = f.astype({c: "float32" for c in f.columns if f[c].dtype == "float64"}).copy()

    keep = (t["eligible"] == 1) & t[["target_10", "target_30", "target_60"]].notna().any(axis=1)
    out = pd.concat([f[keep], t.loc[keep, TARGETS]], axis=1).copy()
    return out


def main():
    p = argparse.ArgumentParser(description="조기 경보 특징 생성")
    p.add_argument("--clean", default=os.path.join(PROJECT_ROOT, "data", "cleaned"))
    p.add_argument("--target", default=os.path.join(PROJECT_ROOT, "data", "target"))
    p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "data", "features"))
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    # 분할 전체를 한 파일로 저장할 경우, Training이 500MB라 16GB 노트북에서 다루기 버거워 설비별로 나눠 저장
    total = {}
    for f in sorted(glob.glob(os.path.join(args.target, "*", "*", "*.parquet"))):
        rel = os.path.relpath(f, args.target)
        split, facility = rel.split(os.sep)[:2]
        out = make_features(pd.read_parquet(os.path.join(args.clean, rel)), pd.read_parquet(f))
        out.insert(0, "dev", int(os.path.basename(f)[:-8]))
        out.insert(1, "facility", facility)
        path = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_parquet(path)
        n, pos = total.get(split, (0, 0))
        total[split] = (n + len(out), pos + int((out["target_30"] == 1).sum()))
    for split, (n, pos) in total.items():
        print(f"{split}: {n:,}행, 30분 양성 {pos:,}건")
    print("완료:", args.dst)

if __name__ == "__main__":
    main()
