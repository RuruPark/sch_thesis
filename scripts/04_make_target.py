# -*- coding: utf-8 -*-
# 조기 경보 정답 라벨: 지금 정상으로 운전 중인 설비가 N분 안에 전류고조파 이상(주의/경고)으로 바뀌는지
# 입력: data/cleaned  출력: data/target/<split>/<설비종류>/<번호>.parquet, data/target/target_summary.csv
# 기준 근거는 docs/problem-definition.md
#
# 사용법: python scripts/04_make_target.py
import argparse, glob, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = "label_기동_전류평균"
SOH = "label_SOH_전류고조파평균"
# 이상 구간의 48%가 5분 미만. 이걸 다 사건으로 볼 경우, 센서 흔들림을 맞히는 문제가 돼서 5분 이상만 인정
MIN_EVENT_MIN = 5
HORIZONS = [10, 30, 60]


def run_lengths(flag):
    run_id = (flag != flag.shift()).cumsum()
    return run_id, flag.groupby(run_id).transform("size")


def make_target(d):
    out = pd.DataFrame(index=d.index)
    # 전류 > 0으로 자를 경우, Stop 상태의 대기 전류(중앙값 2.1A) 구간이 섞여 고조파 '주의'가 부풀려지는 문제 있음
    running = d[STATE].notna() & (d[STATE] != "Stop") & (d["flag_power_off"] == 0)
    abnormal = running & d[SOH].isin(["주의", "경고"])
    run_id, run_len = run_lengths(abnormal)
    is_start = abnormal & (run_id != run_id.shift())
    event_start = is_start & (run_len >= MIN_EVENT_MIN)

    out["is_running"] = running.astype("int8")
    out["is_abnormal"] = abnormal.astype("int8")
    out["event_start"] = event_start.astype("int8")
    # 예측 대상은 지금 정상인 운전 시점만. 5분 미만 깜빡임 중인 시점은 정상도 이상도 아니라서 뺌
    out["eligible"] = (running & (d[SOH] == "정상")).astype("int8")

    t = pd.Series(np.arange(len(d), dtype="float64"), index=d.index)  # 1분 격자라 행 번호 = 분
    next_event = t.where(event_start).bfill().shift(-1)
    out["minutes_to_event"] = next_event - t

    label_known = d[SOH].notna().astype(int)
    for h in HORIZONS:
        hit = out["minutes_to_event"] <= h
        # 앞으로 h분 중 라벨이 빈 분이 있거나 파일 끝을 넘으면 사건이 없었는지 알 수 없어서 정답을 비워 둠
        known = label_known[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1) == h
        target = pd.Series(np.where(hit, 1.0, np.where(known, 0.0, np.nan)), index=d.index)
        out[f"target_{h}"] = target.where(out["eligible"] == 1)
    return out


def main():
    p = argparse.ArgumentParser(description="조기 경보 정답 라벨 생성")
    p.add_argument("--src", default=os.path.join(PROJECT_ROOT, "data", "cleaned"))
    p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "data", "target"))
    args = p.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)

    rows = []
    for f in sorted(glob.glob(os.path.join(src, "*", "*", "*.parquet"))):
        rel = os.path.relpath(f, src)
        split, facility = rel.split(os.sep)[:2]
        out = make_target(pd.read_parquet(f))
        path = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_parquet(path)
        r = dict(split=split, facility=facility, dev=int(os.path.basename(f)[:-8]),
                 n_rows=len(out), running=int(out.is_running.sum()), eligible=int(out.eligible.sum()),
                 events=int(out.event_start.sum()))
        for h in HORIZONS:
            r[f"pos_{h}"] = int((out[f"target_{h}"] == 1).sum())
            r[f"neg_{h}"] = int((out[f"target_{h}"] == 0).sum())
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(dst, "target_summary.csv"), index=False, encoding="utf-8-sig")
    g = s.groupby("split")[["eligible", "events"] + [f"pos_{h}" for h in HORIZONS] + [f"neg_{h}" for h in HORIZONS]].sum()
    for h in HORIZONS:
        g[f"rate_{h}"] = (g[f"pos_{h}"] / (g[f"pos_{h}"] + g[f"neg_{h}"])).round(4)
    print(g[["eligible", "events"] + [f"rate_{h}" for h in HORIZONS]].to_string())
    print("완료:", dst)


if __name__ == "__main__":
    main()
