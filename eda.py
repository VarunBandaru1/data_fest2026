# =============================================================================
# 01_eda.py  —  DataFest 2026  |  Run section by section with Ctrl+Enter
# =============================================================================
# TIP: In VS Code, highlight a block and press Shift+Enter to run just that
# section in the interactive terminal (make sure Python extension is installed)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── UPDATE this to your actual folder path ────────────────────────────────────
DATA_DIR = Path(".")   # or e.g. Path("/Users/yourname/2026-ASA-DataFest-Data-Files")
# ─────────────────────────────────────────────────────────────────────────────

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", palette="muted")

print("✓ Imports OK")


# =============================================================================
# SECTION 1 — Load small / reference files  (fast, run first)
# =============================================================================

patients    = pd.read_csv(DATA_DIR / "patients.csv",    low_memory=False)
diagnosis   = pd.read_csv(DATA_DIR / "diagnosis.csv",   low_memory=False)
departments = pd.read_csv(DATA_DIR / "departments.csv", low_memory=False)
providers   = pd.read_csv(DATA_DIR / "providers.csv",   low_memory=False)
tiger       = pd.read_csv(DATA_DIR / "tigercensuscodes.csv", low_memory=False)

print("patients       :", patients.shape)
print("diagnosis      :", diagnosis.shape)
print("departments    :", departments.shape)
print("providers      :", providers.shape)
print("tigercensus    :", tiger.shape)

# Quick column check — paste this output to Claude
for name, df in [("patients", patients), ("diagnosis", diagnosis),
                 ("departments", departments), ("providers", providers),
                 ("tiger", tiger)]:
    print(f"\n[{name}] columns: {list(df.columns)}")


# =============================================================================
# SECTION 2 — Load encounters  (1.47 GB — uses optimized dtypes)
# =============================================================================
# If this crashes due to RAM, change NROWS to 1_000_000 to work on a sample

NROWS = None   # set to e.g. 1_000_000 if you run out of memory

enc_dtypes = {
    "EncounterKey":                 "string",
    "PatientDurableKey":            "string",
    "PrimaryDiagnosisKey":          "string",
    "DepartmentKey":                "string",
    "AttendingProviderDurableKey":  "string",
    "DischargeProviderDurableKey":  "string",
    "ProviderDurableKey":           "string",
    "Type":                         "category",
    "VisitType":                    "category",
    "VisitTypeDescription":         "category",
    "AdmissionSource":              "category",
    "AdmissionType":                "category",
}

encounters = pd.read_csv(
    DATA_DIR / "encounters.csv",
    dtype=enc_dtypes,
    parse_dates=["Date"],
    low_memory=False,
    nrows=NROWS
)

print("\nencounters :", encounters.shape)
print("Columns    :", list(encounters.columns))
print("\nDate range :", encounters["Date"].min(), "→", encounters["Date"].max())
print("\nNull % (encounters):")
print((encounters.isnull().mean() * 100).round(1).to_string())


# =============================================================================
# SECTION 3 — Load social_determinants  (345 MB)
# =============================================================================

sd_dtypes = {
    "EncounterKey":      "string",
    "PatientDurableKey": "string",
    "Domain":            "category",
    "DisplayName":       "string",
    "AnswerText":        "string",
}

soc_det = pd.read_csv(
    DATA_DIR / "social_determinants.csv",
    dtype=sd_dtypes,
    low_memory=False
)

print("\nsoc_det :", soc_det.shape)
print("Columns :", list(soc_det.columns))
print("\nDomain value counts:")
print(soc_det["Domain"].value_counts().to_string())


# =============================================================================
# SECTION 4 — Key distributions  (paste printed output to Claude)
# =============================================================================

# --- 4a. Encounter type breakdown ---
print("\n\n[Encounter Types]")
print(encounters["Type"].value_counts().to_string())

# --- 4b. Visit type description (higher level) ---
print("\n\n[Top 20 VisitTypeDescription]")
print(encounters["VisitTypeDescription"].value_counts().head(20).to_string())

# --- 4c. Boolean flags ---
flag_cols = [c for c in encounters.columns if c.startswith("Is")]
print("\n\n[Boolean flags — % True]")
print((encounters[flag_cols].apply(lambda x: pd.to_numeric(x, errors="coerce")).mean() * 100).round(2).to_string())

# --- 4d. Encounters per patient ---
enc_per_patient = encounters.groupby("PatientDurableKey").size()
print("\n\n[Encounters per patient]")
print(enc_per_patient.describe().round(1))
print("Patients with 1 encounter :", (enc_per_patient == 1).sum())
print("Patients with 10+ encounters:", (enc_per_patient >= 10).sum())
print("Patients with 50+ encounters:", (enc_per_patient >= 50).sum())

# --- 4e. Patient demographics ---
print("\n\n[Patient birth year bins]")
print(patients["PatientBirthYearBin"].value_counts().sort_index().to_string())

print("\n\n[Sex assigned at birth]")
print(patients["SexAssignedAtBirth"].value_counts().to_string())

print("\n\n[VitalStatus]")
print(patients["VitalStatus"].value_counts().to_string())

print("\n\n[FirstRace top 10]")
print(patients["FirstRace"].value_counts().head(10).to_string())

print("\n\n[OmbEthnicity]")
print(patients["OmbEthnicity"].value_counts().to_string())

print("\n\n[MyChartStatus]")
print(patients["MyChartStatus"].value_counts().to_string())

print("\n\n[SmokingStatus]")
print(patients["SmokingStatus"].value_counts().to_string())

# --- 4f. Diagnosis overview ---
print("\n\n[Top 20 GroupName in diagnosis]")
print(diagnosis["GroupName"].value_counts().head(20).to_string())

print("\n\n[Null % in diagnosis]")
print((diagnosis.isnull().mean() * 100).round(1).to_string())

# --- 4g. SDOH coverage ---
n_total   = patients["DurableKey"].nunique()
n_sdoh    = soc_det["PatientDurableKey"].nunique()
print(f"\n\n[SDOH Coverage]")
print(f"Patients with any SDOH data: {n_sdoh:,} / {n_total:,}  ({100*n_sdoh/n_total:.1f}%)")

print("\n[SDOH — unique patients per domain]")
print(soc_det.groupby("Domain")["PatientDurableKey"].nunique().sort_values(ascending=False).to_string())


# =============================================================================
# SECTION 5 — Quick plots  (visual sanity check)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("DataFest 2026 — EDA Overview", fontsize=14)

# Encounter volume over time
monthly = encounters.groupby(encounters["Date"].dt.to_period("M")).size()
monthly.index = monthly.index.to_timestamp()
axes[0,0].plot(monthly.index, monthly.values, lw=1.5)
axes[0,0].set_title("Monthly Encounter Volume")
axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}K"))

# Encounter type
tc = encounters["Type"].value_counts().head(10)
tc.sort_values().plot(kind="barh", ax=axes[0,1])
axes[0,1].set_title("Top 10 Encounter Types")

# Encounters per patient (capped)
enc_per_patient.clip(upper=30).value_counts().sort_index().plot(
    kind="bar", ax=axes[1,0])
axes[1,0].set_title("Encounters per Patient (capped at 30)")
axes[1,0].tick_params(axis="x", rotation=45)

# SDOH domain coverage
soc_det.groupby("Domain")["PatientDurableKey"].nunique().sort_values().plot(
    kind="barh", ax=axes[1,1])
axes[1,1].set_title("Unique Patients per SDOH Domain")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n✓ Saved eda_overview.png")

print("\n\n=== EDA COMPLETE — paste all printed output above to Claude ===")