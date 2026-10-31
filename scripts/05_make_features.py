# -*- coding: utf-8 -*-
# 조기 경보 모델 입력 특징: 최근 10/30/60분 통계, 상간 불균형, 최근 이상 이력, 시간대
# 입력: data/cleaned, data/target  출력: data/features/<split>/<설비종류>/<번호>.parquet (예측 대상 시점만)
# 모든 특징은 그 시점까지의 값만 씀. 미래 값을 섞을 경우, 검증 성능이 실제보다 높게 나오는 문제 있음
#
# --extra: 상별 고조파, 경계 근접, 하루 이력, 고조파 전류 특징 추가 (data/features_v2용, case-studies/feature-extension.md)
# --phase: 상별 전류·전력·역률·전압고조파의 최대/최소 상, 상전압·선간전압 불균형, 누적전력량 증가분 추가 (data/features_v3용)
#
# 사용법: python scripts/05_make_features.py [--target data/target_vh --dst data/features_vh] [--extra --dst data/features_v2]
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


def extra_features(d, t, s):
    h = s["전류고조파평균"]
    f = {}
    # 라벨은 상 평균 기준이라, 한 상이 먼저 경계를 넘는 경우를 평균만으로는 늦게 봄
    ph = d[[f"{p}상전류고조파" for p in "RST"]].astype("float32")
    hmax = ph.max(axis=1)
    f["전류고조파최대상"] = hmax
    f["전류고조파최대상_max30"] = hmax.rolling(30, min_periods=15).max()
    f["전류고조파최대상_max60"] = hmax.rolling(60, min_periods=30).max()
    f["전류고조파최대상_초과"] = hmax - h
    # 주의 경계 30 기준. 라벨이 이 경계로 붙어서 거리와 접근 속도를 직접 줌
    f["경계까지"] = 30 - h
    for lo in (20, 25):
        f[f"고조파{lo}이상_분60"] = (h >= lo).astype("float32").rolling(60, min_periods=30).sum()
    f["전류고조파_기울기10"] = (h.rolling(5, min_periods=3).mean() - h.shift(5).rolling(5, min_periods=3).mean()) / 5
    for span in (5, 15):
        f[f"전류고조파_ewm{span}"] = h.ewm(span=span, min_periods=span).mean()
    # 하루치 이력. 마지막이상후_분은 처음 보는 설비에서 이력이 짧아 비는 경우가 많음
    abn = t["is_abnormal"].astype("float32")
    rise = ((abn == 1) & (abn.shift(fill_value=0) == 0)).astype("float32")  # event_start는 5분 뒤에야 확정돼서 시작 순간만 셈
    f["이상시작수_1440"] = rise.rolling(1440, min_periods=360).sum()
    f["이상비율_1440"] = abn.rolling(1440, min_periods=360).mean()
    f["전류고조파_q90_1440"] = h.rolling(1440, min_periods=360).quantile(0.9)
    f["운전비율_1440"] = t["is_running"].astype("float32").rolling(1440, min_periods=360).mean()
    # 고조파율은 전류가 작을 때 부풀려짐(docs/problem-definition.md). 고조파 전류 크기와 하루 최대 대비 부하를 같이 줌
    hc = s["전류평균"] * h / 100
    f["고조파전류"] = hc
    f["고조파전류_max60"] = hc.rolling(60, min_periods=30).max()
    f["부하율_1440"] = s["전류평균"] / s["전류평균"].rolling(1440, min_periods=360).max().where(lambda v: v > 0)
    return pd.DataFrame(f, index=d.index)


PHASE_ITEMS = ["전류", "유효전력", "무효전력", "역률", "전압고조파"]


def phase_features(d):
    # 지금 특징은 상 평균 중심이라 한 상만 틀어지는 경우를 못 봄. 원본 35개 항목 중 안 쓰던 상별 값을 상 사이 최대·최소로 요약
    f = {}
    for item in PHASE_ITEMS:
        x = d[[f"{p}상{item}" for p in "RST"]].astype("float32")
        hi, lo = x.max(axis=1), x.min(axis=1)
        f[f"{item}_최대상"], f[f"{item}_최소상"] = hi, lo
        f[f"{item}_상차"] = hi - lo
        f[f"{item}_상차_mean60"] = (hi - lo).rolling(60, min_periods=30).mean()
    f["상전압불균형"] = imbalance(d, "전압")
    f["선간전압불균형"] = imbalance(d, "선간전압")
    # 누적전력량은 설비마다 시작값이 달라 그대로 쓸 수 없어서 60분 증가분만 씀. 음수는 계량기 초기화로 보고 비움
    e = d["누적전력량"].astype("float32").diff(60)
    f["전력량_diff60"] = e.where(e >= 0)
    return pd.DataFrame(f, index=d.index)


def make_features(d, t, extra=False, phase=False):
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
    if extra:
        f = pd.concat([f, extra_features(d, t, s)], axis=1)
    if phase:
        f = pd.concat([f, phase_features(d)], axis=1)

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
    p.add_argument("--extra", action="store_true", help="보강 특징 추가")
    p.add_argument("--phase", action="store_true", help="상별 측정 항목 특징 추가")
    args = p.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    # 분할 전체를 한 파일로 저장할 경우, Training이 500MB라 16GB 노트북에서 다루기 버거워 설비별로 나눠 저장
    total = {}
    for f in sorted(glob.glob(os.path.join(args.target, "*", "*", "*.parquet"))):
        rel = os.path.relpath(f, args.target)
        split, facility = rel.split(os.sep)[:2]
        out = make_features(pd.read_parquet(os.path.join(args.clean, rel)), pd.read_parquet(f), args.extra, args.phase)
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
