# -*- coding: utf-8 -*-
# data/processed의 parquet를 csv.gz로 한 벌 더 저장 (Claude 분석 환경과 공유용)
# 입력: data/processed (--src)  출력: data/processed_csv (--dst)
# 사용법: python scripts/01b_export_csv.py
import argparse, os, glob
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser(description="parquet -> csv.gz 변환")
p.add_argument("--src", default=os.path.join(PROJECT_ROOT, "data", "processed"), help="parquet 폴더")
p.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "data", "processed_csv"), help="csv.gz 출력 폴더")
args = p.parse_args()
SRC, DST = os.path.abspath(args.src), os.path.abspath(args.dst)

files = sorted(glob.glob(os.path.join(SRC, "**", "*.parquet"), recursive=True))
for i, f in enumerate(files, 1):
    out = os.path.join(DST, os.path.relpath(f, SRC)).replace(".parquet", ".csv.gz")
    if os.path.exists(out):
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.read_parquet(f).to_csv(out, encoding="utf-8", compression="gzip")
    print(f"({i}/{len(files)}) {os.path.relpath(out, DST)}")
print("완료:", DST)
