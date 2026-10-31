# -*- coding: utf-8 -*-
# 처음 보는 설비 PR-AUC의 설비 단위 부트스트랩 신뢰구간과 모델 간 짝 비교
# 입력: 17번 --save-scores의 extra_fac_<태그>_scores.parquet / _thresholds.csv, 또는 18번 --save-scores의 widen_<태그>_scores.parquet
# 출력: data/results/boot_<태그>_ci.csv, boot_<태그>_diff.csv
# 묶음별 PR-AUC가 0.445~0.871로 흔들려 0.01~0.03 차이를 말로만 "참고 수준"이라 적을 경우, 어디까지가 확실한 결론인지 알 수 없는 문제 있음
# 행이 아니라 설비를 다시 뽑음. 같은 설비의 1분 행은 서로 강하게 이어져 있어 행 단위로 뽑으면 구간이 지나치게 좁아짐
#
# 사용법: python scripts/19_unit_bootstrap.py --scores data/results/extra_fac_v2cmp_scores.parquet [--n 1000] [--tag v2cmp]
#         python scripts/19_unit_bootstrap.py --scores data/results/widen_v2s_scores.parquet --by facility --tag v2s
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate import RESULTS_DIR, TARGET

META = {"time", "dev", "fold", "facility", "minutes_to_event", "target_10", "target_30", "target_60"}


def model_columns(sc, thr_path):
    # 17번은 열 이름이 score_i라 thresholds.csv에서 모델명을 찾음. 18번은 열 이름이 모델명
    if thr_path and os.path.exists(thr_path):
        m = pd.read_csv(thr_path).drop_duplicates("column").set_index("column")["model"]
        return {c: m[c] for c in sc.columns if c in m.index}
    return {c: c for c in sc.columns if c not in META}


def boot(sc, cols, target, n, seed):
    # 설비마다 행 위치를 묶어 두고, 뽑힌 설비의 행을 이어 붙여 PR-AUC를 다시 계산
    rng = np.random.default_rng(seed)
    y = sc[target].to_numpy()
    s = {c: sc[c].to_numpy() for c in cols}
    groups = [np.flatnonzero(m) for m in (sc["dev"].to_numpy()[:, None] == sc["dev"].unique()[None, :]).T]
    out = {c: np.full(n, np.nan) for c in cols}
    for b in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() == 0:
            continue
        for c in cols:
            out[c][b] = average_precision_score(y[idx], s[c][idx])
    return out


def main():
    p = argparse.ArgumentParser(description="설비 단위 부트스트랩 신뢰구간")
    p.add_argument("--scores", required=True)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--by", default="", help="facility를 주면 설비종류마다 따로 계산 (18번 결과)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dst", default=RESULTS_DIR)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    sc = pd.read_parquet(args.scores)
    cols = model_columns(sc, args.scores.replace("_scores.parquet", "_thresholds.csv"))
    prefix = os.path.join(args.dst, "boot" + (f"_{args.tag}" if args.tag else ""))

    ci_rows, diff_rows = [], []
    groups = sc.groupby(args.by) if args.by else [("전체", sc)]
    for g, d in groups:
        # 같은 설비 번호가 설비종류마다 겹칠 수 있어 종류 안에서만 뽑음
        b = boot(d, list(cols), args.target, args.n, args.seed)
        y = d[args.target].to_numpy()
        for c, name in cols.items():
            v = b[c][~np.isnan(b[c])]
            ci_rows.append(dict(group=g, model=name, units=d["dev"].nunique(), pr_auc=average_precision_score(y, d[c]),
                                ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5), n_boot=len(v)))
        # 짝 비교: 같은 부트스트랩 표본에서 두 모델 차이. 0을 포함하지 않으면 그 방향이 설비 구성과 관계없이 유지된다고 봄
        names = list(cols)
        for i, a in enumerate(names):
            for c in names[i + 1:]:
                dv = b[c] - b[a]
                dv = dv[~np.isnan(dv)]
                diff_rows.append(dict(group=g, model_a=cols[a], model_b=cols[c],
                                      diff=average_precision_score(y, d[c]) - average_precision_score(y, d[a]),
                                      ci_low=np.percentile(dv, 2.5), ci_high=np.percentile(dv, 97.5),
                                      p_b_better=(dv > 0).mean()))
        print(f"{g} 완료 ({d['dev'].nunique()}대)")

    ci, diff = pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)
    ci.to_csv(prefix + "_ci.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(prefix + "_diff.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(ci.round(3).to_string(index=False))
        print(diff.round(3).to_string(index=False))
    print("완료:", prefix)


if __name__ == "__main__":
    main()
