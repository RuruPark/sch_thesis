# -*- coding: utf-8 -*-
# 06번 이후 실험에서 같이 쓰는 데이터 읽기, 경보 임계값 선택, 지표 계산
# 지표 정의는 docs/metrics.md
import glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
TARGET = "target_30"
NON_FEATURES = ["dev", "facility", "target_10", "target_30", "target_60", "minutes_to_event"]
HIST = "마지막이상후_분"


def load(split, src=FEATURES_DIR):
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(src, split, "*", "*.parquet")))]
    d = pd.concat(parts)
    return d[d[TARGET].notna()].copy()


def feature_columns(d):
    return [c for c in d.columns if c not in NON_FEATURES]


def split_fit_calib(d, frac=0.8):
    # Training 안에서 임계값을 고를 경우, 학습에 쓴 행으로 고르면 트리 모델은 점수가 과하게 높아 임계값이 어긋나는 문제 있어 뒤 20%를 떼어 둠
    rank = d.groupby("dev").cumcount() / d.groupby("dev")["dev"].transform("size")
    return d[rank < frac], d[rank >= frac]


def add_history_flag(d, fill):
    # 이상 이력이 없는 시점(약 70%)을 중앙값으로 채울 경우, "최근에 이상이 있었다"와 구분이 안 되는 문제 있어 따로 표시하고 큰 값으로 채움
    d = d.copy()
    d["이상이력없음"] = d[HIST].isna().astype("float32")
    d[HIST] = d[HIST].fillna(fill)
    return d


def clean(X):
    return X.replace([np.inf, -np.inf], np.nan)


def best_threshold(y, score):
    # Validation을 보고 임계값을 고를 경우, 성능이 부풀려지는 문제 있어 학습 쪽 데이터에서 F1 최대 지점으로 고정
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return t[np.argmax(f1[:-1])]


def run_ids(d, flag):
    # 같은 설비에서 1분 간격으로 이어진 flag 구간마다 번호. flag가 아닌 행은 NaN
    t = d.index.to_series().reset_index(drop=True)
    dev = d["dev"].reset_index(drop=True)
    flag = pd.Series(flag)
    new = (dev != dev.shift()) | (t - t.shift() != pd.Timedelta(minutes=1)) | ~flag.shift(fill_value=False)
    return (new & flag).cumsum().where(flag)


def sustained(d, alarm, n):
    # 경보가 n분 연속 이어진 시점부터만 알림. 1분짜리 튐에 반응할 경우, 오경보가 쌓이는 문제 있음
    if n <= 1:
        return alarm
    rid = run_ids(d, alarm)
    pos_in_run = rid.groupby(rid).cumcount() + 1
    return (pos_in_run.reindex(rid.index) >= n).to_numpy() & alarm


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
    rid = run_ids(d, alarm)
    false_runs = int((~pd.Series(y == 1).groupby(rid).any()).sum())

    return dict(precision=precision, recall=recall, f1=f1,
                events=len(ev), event_recall=caught.mean(),
                lead_median=ev[caught].median(), false_alarm_per_day=false_runs / (len(d) / 1440))
