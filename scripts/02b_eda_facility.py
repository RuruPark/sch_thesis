# -*- coding: utf-8 -*-
# 설비종류 10종으로 늘린 1차 전처리 결과를 설비종류별로 점검. 펌프 기준으로 정한 라벨 경계와 2차 전처리 기준이 그대로 통하는지 확인용
# 입력: data/processed  출력: data/eda/facility_*.csv
# 02_eda.py는 전체를 한 번에 올려서 461대(약 1,900만 행)에서는 16GB 메모리가 모자람 → 파일 하나씩 읽어 집계만 누적
#
# 사용법: python scripts/02b_eda_facility.py
import argparse, glob, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE_I = ["R상전류", "S상전류", "T상전류"]
PHASE_V = ["R상전압", "S상전압", "T상전압"]
STATE = "label_기동_전류평균"
SOH = {"label_SOH_역률평균": "역률평균",
       "label_SOH_전류고조파평균": "전류고조파평균",
       "label_SOH_전압고조파평균": "전압고조파평균"}
# 02_eda.py에서 펌프·공조 운전 구간으로 잡은 경계
SOH_CUTS = {"역률평균": (0.8, 0.6), "전류고조파평균": (30, 60), "전압고조파평균": (3, 5)}
LEVELS = ["정상", "주의", "경고"]
QS = [0.02, 0.5, 0.98]


def soh_rule(col, x):
    hi, lo = SOH_CUTS[col]
    if col == "역률평균":
        return np.select([x >= hi, x >= lo], ["정상", "주의"], "경고")
    return np.select([x < hi, x < lo], ["정상", "주의"], "경고")


def abnormal_runs(lab, running):
    # 04_make_target.py와 같은 기준: 운전 중 정상 다음에 시작해 5분 이상 이어진 주의/경고
    abn = running & np.isin(lab, ["주의", "경고"])
    norm = running & (lab == "정상")
    starts, n, i = 0, len(lab), 1
    while i < n:
        if abn[i] and norm[i - 1]:
            j = i
            while j < n and abn[j]:
                j += 1
            starts += (j - i) >= 5
            i = j
        else:
            i += 1
    return starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(PROJECT_ROOT, "data", "processed"))
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "eda"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev_rows, lab_rows, val_samples = [], [], {}
    files = sorted(glob.glob(os.path.join(args.src, "*", "*", "*.parquet")))
    for k, f in enumerate(files, 1):
        split, facility = f.split(os.sep)[-3:-1]
        d = pd.read_parquet(f)
        running = (d[STATE] != "Stop").to_numpy() & d[STATE].notna().to_numpy()
        power_off = (d["상전압평균"] < 50).to_numpy()
        r = dict(split=split, facility=facility, dev=int(os.path.basename(f)[:-8]), n_rows=len(d),
                 running_ratio=running.mean(),
                 run_current_median=d.loc[running, "전류평균"].median(),
                 stop_current_median=d.loc[~running & ~power_off, "전류평균"].median(),
                 clip_rows=int(((d[PHASE_I] == 512) | (d[PHASE_I] >= 4095)).any(axis=1).sum()),
                 volt_spike_rows=int((d[PHASE_V] > 500).any(axis=1).sum()),
                 freq_out_ratio=(~power_off & d["주파수"].notna() & ~d["주파수"].between(55, 65)).mean(),
                 neg_power_ratio=(d["유효전력평균"] < 0).mean(),
                 events_5min=abnormal_runs(d["label_SOH_전류고조파평균"].to_numpy(), running))
        for lab, col in SOH.items():
            x, y = d.loc[running, col].to_numpy(), d.loc[running, lab].to_numpy()
            ok = ~pd.isna(y)
            r[f"rule_match_{col}"] = (soh_rule(col, x[ok]) == y[ok]).mean() if ok.any() else np.nan
            r[f"abn_ratio_{col}"] = np.isin(y[ok], ["주의", "경고"]).mean() if ok.any() else np.nan
            for v in LEVELS:
                m = y == v
                lab_rows.append(dict(split=split, facility=facility, label=lab, value=v, n=int(m.sum())))
                if m.any():
                    # 경계 확인용 분위수는 설비종류별 표본을 모아서 계산
                    s = val_samples.setdefault((facility, col, v), [])
                    s.append(x[m][:: max(1, m.sum() // 2000)])
        dev_rows.append(r)
        if k % 50 == 0:
            print(f"{k}/{len(files)}")

    dev = pd.DataFrame(dev_rows)
    dev.round(4).to_csv(os.path.join(args.out, "facility_device_summary.csv"), index=False, encoding="utf-8-sig")

    agg = dev.groupby(["facility", "split"]).agg(
        devices=("dev", "size"), rows=("n_rows", "sum"), running_ratio=("running_ratio", "mean"),
        run_current_median=("run_current_median", "median"), stop_current_median=("stop_current_median", "median"),
        clip_rows=("clip_rows", "sum"), volt_spike_rows=("volt_spike_rows", "sum"),
        freq_out_devices=("freq_out_ratio", lambda s: int((s > 0.01).sum())),
        neg_power_devices=("neg_power_ratio", lambda s: int((s > 0.01).sum())),
        events_5min=("events_5min", "sum"), devices_with_event=("events_5min", lambda s: int((s > 0).sum())),
        **{f"rule_match_{c}": (f"rule_match_{c}", "mean") for c in SOH.values()},
        **{f"abn_ratio_{c}": (f"abn_ratio_{c}", "mean") for c in SOH.values()})
    agg.round(4).to_csv(os.path.join(args.out, "facility_summary.csv"), encoding="utf-8-sig")

    lab = pd.DataFrame(lab_rows).groupby(["facility", "split", "label", "value"])["n"].sum().reset_index()
    lab.to_csv(os.path.join(args.out, "facility_label_distribution.csv"), index=False, encoding="utf-8-sig")

    q = [dict(facility=fa, item=col, label=v, n=sum(map(len, s)),
              **{f"q{int(p * 100):02d}": np.nanquantile(np.concatenate(s), p) for p in QS})
         for (fa, col, v), s in sorted(val_samples.items())]
    pd.DataFrame(q).round(3).to_csv(os.path.join(args.out, "facility_label_quantiles.csv"),
                                    index=False, encoding="utf-8-sig")
    print(agg[["devices", "rows", "running_ratio", "events_5min", "devices_with_event",
               "rule_match_전류고조파평균"]].to_string())


if __name__ == "__main__":
    main()
