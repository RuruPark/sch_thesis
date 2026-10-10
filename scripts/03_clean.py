# -*- coding: utf-8 -*-
# 2차 전처리: 설비별 parquet를 1분 격자에 맞추고 측정 상한값, 전압 튐, 주파수 이상을 결측으로 바꾼 뒤 짧은 결측만 보간
# 입력: data/processed  출력: data/cleaned/<split>/<설비종류>/<번호>.parquet, data/cleaned/clean_log.csv
# 기준값은 02_eda.py 결과(data/eda/*.csv)에서 정함. 근거는 docs/preprocessing.md 2차 전처리 절
#
# 사용법: python scripts/03_clean.py
import argparse, glob, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASES = ["R", "S", "T"]
PHASE_I = [f"{p}상전류" for p in PHASES]
PHASE_V = [f"{p}상전압" for p in PHASES]
LINE_V = [f"{p}상선간전압" for p in PHASES]

# 512, 4095.94는 전류 센서 상한. 기동 순간 돌입전류가 이 값으로 잘려 들어와서 실제 크기를 알 수 없음
CLIP_VALUES = (512.0,)
CLIP_FLOOR = 4095.0
# 정격 220/380/440V 설비뿐인데 상전압 940~999V가 한두 분씩 튐(484, 486번). 500V 넘으면 측정 오류로 봄
MAX_PHASE_V, MAX_LINE_V = 500, 800
# 548번(인버터1동력) Training 1/19~1/28은 주파수 167Hz, 상전압 97V인데 유효전력이 0에 가까움. 전원이 들어온 상태에서 60Hz ±5 밖이면 계측 오류로 봄
FREQ_RANGE = (55, 65)
# 299~301번은 전원이 꺼진 동안 상전압 7V, 주파수 20Hz 같은 잔류값이 찍힘. 0만 꺼짐으로 볼 경우, 이 구간이 주파수 이상으로 잘못 걸리는 문제 있음
POWER_OFF_V = 50
# 1시간 넘게 끊긴 곳이 73곳(최대 61시간). 이걸 채울 경우, 없는 운전 패턴이 생기는 문제 있어 5분 이하만 보간
INTERP_LIMIT = 5
# Validation 80번은 전체가 전압 0(전원 꺼짐)이라 분석에 쓸 값이 없음
MAX_POWER_OFF = 0.95


def to_minute_grid(d):
    # 측정 초가 설비마다 달라(:00, :34 등) 분 단위로 내리고, 같은 분에 2행이면 먼저 들어온 값 사용
    d = d.copy()
    d.index = d.index.floor("min")
    d = d[~d.index.duplicated(keep="first")]
    grid = pd.date_range(d.index.min(), d.index.max(), freq="min", name="TIMESTAMP")
    flag_missing = ~grid.isin(d.index)
    d = d.reindex(grid)
    d["flag_missing"] = flag_missing.astype("int8")
    return d


def mask_bad_values(d, num_cols):
    clip = d[PHASE_I].isin(CLIP_VALUES) | (d[PHASE_I] >= CLIP_FLOOR)
    clip_row = clip.any(axis=1)
    d[PHASE_I] = d[PHASE_I].mask(clip)
    d.loc[clip_row, "전류평균"] = np.nan

    v_bad = ((d[PHASE_V + ["상전압평균"]] > MAX_PHASE_V).any(axis=1)
             | (d[LINE_V + ["선간전압평균"]] > MAX_LINE_V).any(axis=1))
    volt_cols = PHASE_V + LINE_V + ["상전압평균", "선간전압평균"]
    d.loc[v_bad, volt_cols] = np.nan

    f = d["주파수"]
    f_bad = (d["상전압평균"] >= POWER_OFF_V) & f.notna() & ~f.between(*FREQ_RANGE)
    d.loc[f_bad, num_cols] = np.nan

    d["flag_clip"] = clip_row.astype("int8")
    d["flag_invalid"] = (v_bad | f_bad).astype("int8")
    return int(clip_row.sum()), int(v_bad.sum()), int(f_bad.sum())


def short_gap(isna, limit):
    # pandas interpolate(limit=5)는 긴 결측도 앞 5칸을 채워버려서, 결측 길이를 직접 세서 5칸 이하만 고름
    run_id = (~isna).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    return isna & (run_len <= limit)


def fill_short(d, cols, how):
    for c in cols:
        isna = d[c].isna()
        if not isna.any():
            continue
        filled = d[c].interpolate(method="linear", limit_area="inside") if how == "linear" else d[c].ffill()
        d[c] = d[c].where(~short_gap(isna, INTERP_LIMIT), filled)


def clean_device(d):
    label_cols = [c for c in d.columns if c.startswith("label_")]
    num_cols = [c for c in d.columns if c not in label_cols]
    n_in = len(d)
    d = to_minute_grid(d)
    n_clip, n_volt, n_freq = mask_bad_values(d, num_cols)
    nan_before = int(d[num_cols].isna().any(axis=1).sum())

    fill_short(d, num_cols, "linear")
    # 기동 상태는 몇 분 안에 잘 안 바뀌어서 앞 값으로 채움. 5분 넘게 끊긴 곳은 라벨도 비워 둠
    fill_short(d, label_cols, "ffill")

    # 전원이 꺼진 시간. 값 자체는 실제 상태라 지우지 않고 표시만 함
    d["flag_power_off"] = (d["상전압평균"] < POWER_OFF_V).astype("int8")
    nan_after = int(d[num_cols].isna().any(axis=1).sum())
    stats = dict(n_in=n_in, n_out=len(d), added_rows=int(d["flag_missing"].sum()),
                 clip_rows=n_clip, volt_spike_rows=n_volt, freq_bad_rows=n_freq,
                 nan_rows_before_interp=nan_before, nan_rows_after_interp=nan_after,
                 power_off_ratio=round(float(d["flag_power_off"].mean()), 4))
    return d, stats


def main():
    p = argparse.ArgumentParser(description="설비별 parquet 2차 전처리")
    p.add_argument("--src", default=os.path.join(PROJECT_ROOT, "data", "processed"))
    p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "data", "cleaned"))
    args = p.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)

    files = sorted(glob.glob(os.path.join(src, "*", "*", "*.parquet")))
    logs = []
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, src)
        split, facility = rel.split(os.sep)[:2]
        dev = os.path.basename(f)[:-8]
        d, stats = clean_device(pd.read_parquet(f))
        stats.update(split=split, facility=facility, dev=int(dev))
        if stats["power_off_ratio"] >= MAX_POWER_OFF:
            stats["excluded"] = "전원 꺼짐 95% 이상"
            print(f"({i}/{len(files)}) {rel}: 제외 (전원 꺼짐 {stats['power_off_ratio']:.0%})")
        else:
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            d.to_parquet(out)
            stats["excluded"] = ""
        logs.append(stats)
    log = pd.DataFrame(logs)
    log = log[["split", "facility", "dev"] + [c for c in log.columns if c not in ("split", "facility", "dev")]]
    log.to_csv(os.path.join(dst, "clean_log.csv"), index=False, encoding="utf-8-sig")
    kept = log[log.excluded == ""]
    print(f"\n완료: {len(kept)}/{len(log)}개 파일, {kept.n_out.sum():,}행 "
          f"(추가된 빈 분 {kept.added_rows.sum():,}, 상한값 {kept.clip_rows.sum()}, "
          f"전압 튐 {kept.volt_spike_rows.sum()}, 주파수 이상 {kept.freq_bad_rows.sum():,}행)")
    print("결과 폴더:", dst)


if __name__ == "__main__":
    main()
