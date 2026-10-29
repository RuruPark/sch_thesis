# -*- coding: utf-8 -*-
# AI-Hub 전력설비 라벨링 JSON → 설비별 parquet (시각 × 측정항목 35개 + 라벨 5개)
# 입력: <원본>/<Training|Validation>/<설비종류>/... (--raw-dir 또는 THESIS_RAW_DIR, 기본 data/raw)
#       폴더가 없는 설비종류는 같은 위치의 [라벨링]<설비종류>.zip 안을 바로 읽음
# 출력: data/processed/<split>/<설비종류>/<번호>.parquet, meta.csv, errors.csv
# 설비 1대의 JSON 5개는 센서값이 같고 라벨만 다름 → 센서값은 한 번만 읽고 라벨만 컬럼으로 붙임
# 이미 만든 parquet는 건너뜀. 중간에 멈춰도 다시 돌리면 이어서 진행
#
# 사용법: python scripts/01_preprocess.py --raw-dir "<원본 데이터 폴더>" [--facility 3.공기압축기 ...]
import argparse, json, os, re, glob, time, zipfile
import pandas as pd

# 경로는 프로젝트 루트(scripts/ 상위) 기준
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.environ.get("THESIS_RAW_DIR", os.path.join(PROJECT_ROOT, "data", "raw"))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS = ["Training", "Validation"]

# 진단 유형/기준 항목 폴더 -> 라벨 컬럼 이름
TASKS = {
    ("1.기동패턴", "1.전류평균"): "label_기동_전류평균",
    ("1.기동패턴", "2.유효전력평균"): "label_기동_유효전력평균",
    ("2.SOH진단", "1.역률평균"): "label_SOH_역률평균",
    ("2.SOH진단", "2.전류고조파평균"): "label_SOH_전류고조파평균",
    ("2.SOH진단", "3.전압고조파평균"): "label_SOH_전압고조파평균",
}
META_KEYS = ["SVC_NAME", "BASE_ITEM", "DEVICE_ID", "DEVICE_BD_ID", "COMPANY_NAME", "ESTABLISH_YEAR",
             "LOCATION", "COMPANY_TYPE", "COMPANY_CAT", "MAJOR_PRODUCT", "CONTRACT_POWER", "BILL_OPTION",
             "facility_name", "facility_type_name", "facility_vendor", "facility_year",
             "facility_capacity", "facility_volt"]


def find_files(split_dir, facility):
    """설비 번호 -> {라벨컬럼: 파일경로}"""
    files = {}
    for (svc, item), col in TASKS.items():
        for f in glob.glob(os.path.join(split_dir, facility, svc, item, "0.Combined", "*.json")):
            m = re.search(r"LabelledData_(\d+)_", os.path.basename(f))
            if m:
                files.setdefault(m.group(1), {})[col] = f
    return files


def find_zip_files(zip_path):
    """zip 안 경로 기준 설비 번호 -> {라벨컬럼: (zip경로, 내부경로)}"""
    files = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    for (svc, item), col in TASKS.items():
        key = f"/{svc}/{item}/0.Combined/"
        for n in names:
            m = re.search(r"LabelledData_(\d+)_", n)
            if key in n and n.endswith(".json") and m:
                files.setdefault(m.group(1), {})[col] = (zip_path, n)
    return files


def load(path):
    # 풀면 zip의 약 31배(설비 8종 Training 약 400GB)라 압축 해제 없이 메모리로만 읽음
    if isinstance(path, tuple):
        with zipfile.ZipFile(path[0]) as zf:
            d = json.loads(zf.read(path[1]))
    else:
        with open(path, encoding="utf-8") as fp:
            d = json.load(fp)
    data = pd.DataFrame(d.pop("data"))
    return d, data


def convert_device(paths):
    wide, meta = None, None
    for col, path in paths.items():
        d, data = load(path)
        # 같은 시각·항목이 중복된 행이 있을 경우, pivot이 실패하는 문제 있어 첫 행만 남김
        data = data.drop_duplicates(["TIMESTAMP", "ITEM_NAME"])
        if wide is None:  # 센서값은 첫 파일에서 한 번만
            meta = {k: d.get(k) for k in META_KEYS}
            wide = data.pivot(index="TIMESTAMP", columns="ITEM_NAME", values="ITEM_VALUE")
        lab = data.groupby("TIMESTAMP")["LABEL_NAME"].first()
        wide[col] = lab.reindex(wide.index)
        del data
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    wide.index.name = "TIMESTAMP"
    wide.columns.name = None
    return wide, meta


def parse_args():
    p = argparse.ArgumentParser(description="AI-Hub 라벨링 JSON -> 설비별 parquet 변환")
    p.add_argument("--raw-dir", default=DEFAULT_RAW,
                   help="Training/, Validation/ 이 들어 있는 원본 데이터 폴더 (기본: $THESIS_RAW_DIR 또는 data/raw)")
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="출력 폴더 (기본: data/processed)")
    p.add_argument("--facility", nargs="*", help="이 설비종류만 처리 (예: 3.공기압축기). 없으면 전부")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow 가 없습니다. 먼저 'pip install pyarrow' 를 실행해 주세요.")
    raw_dir, out_root = os.path.abspath(args.raw_dir), os.path.abspath(args.out_dir)
    if not any(os.path.isdir(os.path.join(raw_dir, s)) for s in SPLITS):
        raise SystemExit(f"원본 데이터 폴더에 Training/ 또는 Validation/ 이 없습니다: {raw_dir}\n"
                         "--raw-dir 또는 환경변수 THESIS_RAW_DIR 로 경로를 지정해 주세요.")
    os.makedirs(out_root, exist_ok=True)
    meta_path, err_path = os.path.join(out_root, "meta.csv"), os.path.join(out_root, "errors.csv")
    metas = pd.read_csv(meta_path).to_dict("records") if os.path.exists(meta_path) else []
    errors = []
    for split in SPLITS:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            continue
        sources = {}
        for name in os.listdir(split_dir):
            m = re.fullmatch(r"\[라벨링\](.+)\.zip", name)
            if m:
                sources[m.group(1)] = os.path.join(split_dir, name)
        for name in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, name)):
                sources[name] = None  # 이미 풀어 둔 폴더가 있으면 폴더를 씀
        for facility in sorted(sources, key=lambda f: int(f.split(".")[0]) if f.split(".")[0].isdigit() else 99):
            if args.facility and facility not in args.facility:
                continue
            zip_path = sources[facility]
            devs = find_zip_files(zip_path) if zip_path else find_files(split_dir, facility)
            out_dir = os.path.join(out_root, split, facility)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n[{split}/{facility}] 설비 {len(devs)}대{' (zip)' if zip_path else ''}")
            for i, (dev, paths) in enumerate(sorted(devs.items(), key=lambda kv: int(kv[0])), 1):
                out = os.path.join(out_dir, f"{dev}.parquet")
                if os.path.exists(out):
                    continue
                t = time.time()
                try:
                    wide, meta = convert_device(paths)
                    wide.to_parquet(out)
                    meta.update(split=split, facility=facility, dev=dev, n_rows=len(wide),
                                start=wide.index.min(), end=wide.index.max(),
                                n_label_files=len(paths))
                    metas.append(meta)
                    pd.DataFrame(metas).to_csv(meta_path, index=False, encoding="utf-8-sig")
                    print(f"  ({i}/{len(devs)}) {dev}: {len(wide)}행, 라벨 {len(paths)}종, {time.time()-t:.0f}초")
                except Exception as e:
                    # 압축 해제가 덜 된 파일이 섞여 있어 실패는 기록만 하고 넘어감
                    errors.append(dict(split=split, facility=facility, dev=dev, error=repr(e)[:300]))
                    pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
                    print(f"  ({i}/{len(devs)}) {dev}: 실패 -> errors.csv 기록 ({type(e).__name__})")
    print("\n완료. 결과 폴더:", out_root)


if __name__ == "__main__":
    main()
