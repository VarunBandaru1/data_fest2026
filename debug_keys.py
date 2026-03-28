# debug_keys.py — run this to figure out why the merge returns 0 rows

import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")

print("Loading diagnosis...")
diagnosis = pd.read_csv(DATA_DIR / "diagnosis.csv", low_memory=False)

print("Loading encounters (first 100K rows only for speed)...")
encounters = pd.read_csv(DATA_DIR / "encounters.csv", low_memory=False, nrows=100_000)

print("\n--- DiagnosisKey in diagnosis ---")
print("dtype       :", diagnosis["DiagnosisKey"].dtype)
print("sample vals :", diagnosis["DiagnosisKey"].dropna().head(10).tolist())
print("has nulls   :", diagnosis["DiagnosisKey"].isna().sum())

print("\n--- PrimaryDiagnosisKey in encounters ---")
print("dtype       :", encounters["PrimaryDiagnosisKey"].dtype)
print("sample vals :", encounters["PrimaryDiagnosisKey"].dropna().head(10).tolist())
print("has nulls   :", encounters["PrimaryDiagnosisKey"].isna().sum())
print("value -1 count:", (encounters["PrimaryDiagnosisKey"].astype(str) == "-1").sum())

# Try to find any overlap after converting both to string
diag_keys = set(diagnosis["DiagnosisKey"].astype(str).unique())
enc_keys  = set(encounters["PrimaryDiagnosisKey"].astype(str).unique())

overlap = diag_keys & enc_keys
print(f"\nUnique DiagnosisKey values    : {len(diag_keys):,}")
print(f"Unique PrimaryDiagnosisKey val: {len(enc_keys):,}")
print(f"Overlap after str conversion  : {len(overlap):,}")
print(f"Sample overlap keys           : {list(overlap)[:10]}")

# Check if one side has .0 suffix (float formatting)
diag_sample = list(diag_keys)[:5]
enc_sample  = list(enc_keys)[:5]
print(f"\nSample diagnosis keys : {diag_sample}")
print(f"Sample encounter keys : {enc_sample}")

# Try stripping .0
diag_keys_clean = set(k.replace(".0","") for k in diag_keys)
enc_keys_clean  = set(k.replace(".0","") for k in enc_keys)
overlap_clean   = diag_keys_clean & enc_keys_clean
print(f"\nOverlap after stripping .0    : {len(overlap_clean):,}")
print(f"Sample overlap (clean)        : {list(overlap_clean)[:10]}")