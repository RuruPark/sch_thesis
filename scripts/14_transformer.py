# -*- coding: utf-8 -*-
# 비교용 딥러닝 모델: 최근 60분 원시 시계열을 직접 보는 작은 Transformer를 08번과 같은 틀(본 적 있는/처음 보는 설비)로 평가
# 입력: data/features (평가 시점과 라벨), data/cleaned, data/target, data/results/unit_split_assign.csv
# 출력: data/results/transformer_metrics.csv, transformer_folds.csv
# 5분씩 묶어 12개 토큰으로 넣는 PatchTST 방식. 1분을 토큰 하나로 할 경우, GPU 없는 4코어에서 학습이 너무 느린 문제 있음
#
# 사용법: python scripts/14_transformer.py
import argparse, importlib, os, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from evaluate import (FACILITY, FEATURES_DIR, HIST, PROJECT_ROOT, RESULTS_DIR, TARGET, add_history_flag,
                      alarm_metrics, best_threshold, load, split_fit_calib)

feat05 = importlib.import_module("05_make_features")

WINDOW, PATCH = 60, 5
SERIES = feat05.BASE + ["전류불균형", "전류고조파불균형"]
# 이력 관련 값은 60분 창 밖 정보라, 빼면 LightGBM과 입력 정보량이 달라져 같이 넣음
SCALARS = ["이상이력없음", "log_마지막이상후_분", "이상비율_60"]
# 매 epoch 양성 전부 + 음성 일부. 음성 전부를 쓸 경우, epoch 하나에 수십 분 걸리는 문제 있음
NEG_PER_EPOCH = 100_000
MAX_EPOCHS, PATIENCE, BATCH = 10, 2, 512


def build_series(split, clean_dir, target_dir, devs):
    # 설비별 분 단위 원시값(11개) + 운전 여부 + 결측 여부. 설비 사이에 창이 걸치지 않게 앞에 WINDOW-1분을 비워 둠
    arrays, offset = [], {}
    pos = 0
    for dev in devs:
        f = os.path.join(clean_dir, split, FACILITY, f"{dev}.parquet")
        d = pd.read_parquet(f)
        run = pd.read_parquet(os.path.join(target_dir, split, FACILITY, f"{dev}.parquet"), columns=["is_running"])
        s = d[feat05.BASE].astype("float32").copy()
        s["전류불균형"] = feat05.imbalance(d, "전류")
        s["전류고조파불균형"] = feat05.imbalance(d, "전류고조파")
        x = s[SERIES].to_numpy("float32")
        pad = np.full((WINDOW - 1, x.shape[1]), np.nan, dtype="float32")
        run_col = np.concatenate([np.zeros(WINDOW - 1), run["is_running"].to_numpy()]).astype("float32")
        arrays.append((np.concatenate([pad, x]), run_col))
        offset[dev] = (pos + WINDOW - 1, d.index)
        pos += len(x) + WINDOW - 1
    raw = np.concatenate([a for a, _ in arrays])
    run = np.concatenate([r for _, r in arrays])
    return raw, run, offset


def positions(d, offset):
    out = np.empty(len(d), dtype="int64")
    devs = d["dev"].to_numpy()
    for dev in np.unique(devs):
        m = devs == dev
        start, idx = offset[dev]
        out[m] = start + idx.get_indexer(d.index[m])
    return out


def scalars(d, fill):
    d = add_history_flag(d, fill)
    return np.stack([d["이상이력없음"].to_numpy(), np.log1p(d[HIST].to_numpy()) / 10,
                     d["이상비율_60"].fillna(0).to_numpy()], axis=1).astype("float32")


def norm_stats(raw):
    # 정규화 기준은 학습 설비의 Training 원시값에서만 계산. 평가 설비 값을 쓸 경우, 처음 보는 설비 평가가 새는 문제 있음
    med = np.nanmedian(raw, axis=0)
    q1, q3 = np.nanpercentile(raw, [25, 75], axis=0)
    return med, np.where(q3 - q1 > 0, q3 - q1, 1).astype("float32")


class Data:
    def __init__(self, raw, run, med, scale):
        z = np.clip((raw - med) / scale, -10, 10)
        missing = np.isnan(raw).any(axis=1, keepdims=True).astype("float32")
        self.x = np.concatenate([np.nan_to_num(z, nan=0.0), run[:, None], missing], axis=1).astype("float32")

    def windows(self, pos):
        return torch.from_numpy(self.x[pos[:, None] + np.arange(-WINDOW + 1, 1)])


class PatchTransformer(nn.Module):
    def __init__(self, n_ch, n_scalar, d=64, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(PATCH * n_ch, d)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW // PATCH, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(d + n_scalar, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, z):
        b = x.shape[0]
        h = self.encoder(self.embed(x.reshape(b, WINDOW // PATCH, -1)) + self.pos)
        return self.head(torch.cat([h.mean(1), z], 1)).squeeze(1)


@torch.no_grad()
def predict(model, data, pos, z):
    model.eval()
    out = [model(data.windows(pos[i:i + 4096]), torch.from_numpy(z[i:i + 4096])) for i in range(0, len(pos), 4096)]
    return torch.cat(out).sigmoid().numpy()


def train(tr, tr_pos, raw_tr, run_tr, unit_mask):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    fit, cal = split_fit_calib(tr)
    fill = float(fit[HIST].max())
    med, scale = norm_stats(raw_tr[unit_mask])
    data = Data(raw_tr, run_tr, med, scale)
    # split_fit_calib와 같은 기준으로 위치도 나눔
    rank = tr.groupby("dev").cumcount() / tr.groupby("dev")["dev"].transform("size")
    is_fit = (rank < 0.8).to_numpy()
    p_fit, p_cal = tr_pos[is_fit], tr_pos[~is_fit]
    z_fit, z_cal = scalars(fit, fill), scalars(cal, fill)
    y_fit, y_cal = fit[TARGET].to_numpy("float32"), cal[TARGET].to_numpy()

    model = PatchTransformer(data.x.shape[1], len(SCALARS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    pos_idx, neg_idx = np.flatnonzero(y_fit == 1), np.flatnonzero(y_fit == 0)
    best, best_state, wait = -1, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        idx = np.concatenate([pos_idx, rng.choice(neg_idx, min(NEG_PER_EPOCH, len(neg_idx)), replace=False)])
        rng.shuffle(idx)
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(data.windows(p_fit[b]), torch.from_numpy(z_fit[b])), torch.from_numpy(y_fit[b]))
            loss.backward()
            opt.step()
        ap = average_precision_score(y_cal, predict(model, data, p_cal, z_cal))
        print(f"  epoch {epoch}: 뒤 20% PR-AUC {ap:.4f}", flush=True)
        if ap > best:
            best, best_state, wait = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    thr = best_threshold(y_cal, predict(model, data, p_cal, z_cal))
    return model, (med, scale), fill, thr


def main():
    p = argparse.ArgumentParser(description="Transformer 비교 모델")
    p.add_argument("--src", default=FEATURES_DIR)
    p.add_argument("--clean", default=os.path.join(PROJECT_ROOT, "data", "cleaned"))
    p.add_argument("--target-dir", default=os.path.join(PROJECT_ROOT, "data", "target"))
    p.add_argument("--dst", default=RESULTS_DIR)
    args = p.parse_args()
    torch.set_num_threads(4)

    tr, va = load("Training", args.src), load("Validation", args.src)
    folds = pd.read_csv(os.path.join(args.dst, "unit_split_assign.csv")).set_index("dev")["fold"]
    n_folds = int(folds.max()) + 1
    raw_tr, run_tr, off_tr = build_series("Training", args.clean, args.target_dir, sorted(tr["dev"].unique()))
    raw_va, run_va, off_va = build_series("Validation", args.clean, args.target_dir, sorted(va["dev"].unique()))
    tr_pos, va_pos = positions(tr, off_tr), positions(va, off_va)
    dev_of_tr = np.concatenate([np.full(len(i) + WINDOW - 1, d) for d, (_, i) in off_tr.items()])
    tr_fold, va_fold = tr["dev"].map(folds).to_numpy(), va["dev"].map(folds).to_numpy()
    print(f"Training {len(tr):,}행, Validation {len(va):,}행, 입력 {WINDOW}분 x {len(SERIES) + 2}채널")

    def fit_and_scorer(keep_units):
        m = tr["dev"].isin(keep_units).to_numpy()
        t0 = time.time()
        model, stats, fill, thr = train(tr[m], tr_pos[m], raw_tr, run_tr, np.isin(dev_of_tr, list(keep_units)))
        data_va = Data(raw_va, run_va, *stats)
        print(f"  학습 {time.time() - t0:.0f}초", flush=True)
        return lambda te_mask: predict(model, data_va, va_pos[te_mask], scalars(va[te_mask], fill)), thr

    all_units = set(tr["dev"].unique())
    print("전체 설비 모델")
    seen = fit_and_scorer(all_units)
    pooled, fold_rows = {}, []
    for k in range(n_folds):
        print(f"묶음 {k}: 이 묶음 설비 빼고 학습")
        unseen = fit_and_scorer(all_units - set(folds[folds == k].index))
        te_mask = va_fold == k
        y = va[TARGET].to_numpy()[te_mask]
        for setting, (scorer, thr) in {"본 적 있는 설비": seen, "처음 보는 설비": unseen}.items():
            s = scorer(te_mask)
            pooled.setdefault(setting, []).append((s, s >= thr))
            fold_rows.append(dict(fold=k, setting=setting, model="Transformer", units=int(va["dev"][te_mask].nunique()),
                                  positives=int(y.sum()), pr_auc=average_precision_score(y, s) if y.sum() else np.nan))

    va_sorted = pd.concat([va[va_fold == k] for k in range(n_folds)])
    y_all = va_sorted[TARGET].to_numpy()
    rows = []
    for setting, parts in pooled.items():
        s = np.concatenate([x[0] for x in parts])
        alarm = np.concatenate([x[1] for x in parts])
        rows.append(dict(setting=setting, model="Transformer", positive_rate=y_all.mean(),
                         pr_auc=average_precision_score(y_all, s), **alarm_metrics(va_sorted, alarm)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.dst, "transformer_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.dst, "transformer_folds.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
        print(pd.DataFrame(fold_rows).pivot_table(index="setting", columns="fold", values="pr_auc").round(3))
    print("완료:", args.dst)


if __name__ == "__main__":
    main()
