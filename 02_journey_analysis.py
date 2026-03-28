# =============================================================================
# 02_journey_analysis.py  —  DataFest 2026
# Questions:
#   1. Do fracture patients experience long gaps between appointments?
#   2. Does home environment (SDOH: transport, food, financial) predict gaps?
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

DATA_DIR = Path(".")
OUT_DIR  = Path("outputs")
LOG_DIR  = Path("logs")
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file   = open(LOG_DIR / "02_journey_analysis.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", palette="muted")
print("✓ Imports OK")


# =============================================================================
# SECTION 1 — Load files
# =============================================================================

print("\nLoading files...")

patients  = pd.read_csv(DATA_DIR / "patients.csv",  low_memory=False)
diagnosis = pd.read_csv(DATA_DIR / "diagnosis.csv", low_memory=False)

enc_dtypes = {
    "EncounterKey":                "string",
    "PatientDurableKey":           "string",
    "PrimaryDiagnosisKey":         "string",
    "DepartmentKey":               "string",
    "AttendingProviderDurableKey": "string",
    "Type":                        "category",
    "VisitType":                   "category",
    "VisitTypeDescription":        "category",
}
encounters = pd.read_csv(
    DATA_DIR / "encounters.csv",
    dtype=enc_dtypes,
    parse_dates=["Date"],
    date_format="mixed",
    low_memory=False
)

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

print("✓ All files loaded")
print(f"  encounters : {encounters.shape}")
print(f"  soc_det    : {soc_det.shape}")


# =============================================================================
# SECTION 2 — Build fracture encounter base
# =============================================================================

fracture_diag = diagnosis[
    diagnosis["GroupName"].str.contains("Fracture", case=False, na=False)
].copy()

print(f"\nFracture diagnosis records  : {len(fracture_diag):,}")
print(f"Unique GroupNames           : {fracture_diag['GroupName'].nunique()}")
print(f"Unique DiagnosisValues      : {fracture_diag['DiagnosisValue'].nunique()}")
print("\nTop 15 fracture GroupNames:")
print(fracture_diag["GroupName"].value_counts().head(15).to_string())

# Fix type mismatch on DiagnosisKey (float64 → string)
encounters["PrimaryDiagnosisKey"] = encounters["PrimaryDiagnosisKey"].astype(str)
fracture_diag["DiagnosisKey"]     = fracture_diag["DiagnosisKey"].astype(float).astype(int).astype(str)

frac_enc = encounters.merge(
    fracture_diag[["DiagnosisKey", "DiagnosisValue", "DiagnosisName",
                   "GroupCode", "GroupName"]],
    left_on="PrimaryDiagnosisKey", right_on="DiagnosisKey",
    how="inner"
)

print(f"\nFracture encounters         : {len(frac_enc):,}")
print(f"Unique patients             : {frac_enc['PatientDurableKey'].nunique():,}")


# =============================================================================
# SECTION 3 — Build journey-level summary
# =============================================================================

frac_enc = frac_enc.sort_values(["PatientDurableKey", "DiagnosisValue", "Date"])

frac_enc["GapDays"] = (
    frac_enc
    .groupby(["PatientDurableKey", "DiagnosisValue"])["Date"]
    .diff()
    .dt.days
)

journey = (
    frac_enc
    .groupby(["PatientDurableKey", "DiagnosisValue"])
    .agg(
        FirstVisit  = ("Date",               "min"),
        LastVisit   = ("Date",               "max"),
        NumVisits   = ("EncounterKey",        "count"),
        AvgGapDays  = ("GapDays",             "mean"),
        MaxGapDays  = ("GapDays",             "max"),
        HasED       = ("IsEdVisit",           "max"),
        HasHospital = ("IsHospitalAdmission", "max"),
        GroupName   = ("GroupName",           "first"),
        GroupCode   = ("GroupCode",           "first"),
    )
    .reset_index()
)

journey["JourneyDays"]    = (journey["LastVisit"] - journey["FirstVisit"]).dt.days
journey["HasLongGap"]     = journey["MaxGapDays"] >= 60
journey["HasVeryLongGap"] = journey["MaxGapDays"] >= 180

print(f"\n--- Journey Summary ---")
print(f"Total fracture journeys     : {len(journey):,}")
print(f"Unique patients             : {journey['PatientDurableKey'].nunique():,}")
print(f"\nVisits per journey:")
print(journey["NumVisits"].describe().round(1))
print(f"\nJourney length (days):")
print(journey["JourneyDays"].describe().round(1))
print(f"\nMax gap days:")
print(journey["MaxGapDays"].describe().round(1))
print(f"\nJourneys with 60+ day gap  : {journey['HasLongGap'].sum():,}  "
      f"({100*journey['HasLongGap'].mean():.1f}%)")
print(f"Journeys with 180+ day gap : {journey['HasVeryLongGap'].sum():,}  "
      f"({100*journey['HasVeryLongGap'].mean():.1f}%)")
print(f"\nJourneys with ED visit     : {journey['HasED'].sum():,}  "
      f"({100*journey['HasED'].mean():.1f}%)")

print("\nGap distribution by fracture type (GroupName):")
print(journey.groupby("GroupName")["MaxGapDays"]
      .agg(["mean", "median", "count"]).sort_values("median", ascending=False).round(1).to_string())


# =============================================================================
# SECTION 4 — Add patient demographics
# =============================================================================

# Fix type mismatch on PatientDurableKey
patients["DurableKey"]           = patients["DurableKey"].astype(str)
journey["PatientDurableKey"]     = journey["PatientDurableKey"].astype(str)

journey_pat = journey.merge(
    patients[[
        "DurableKey", "PatientBirthYearBin", "FirstRace", "OmbEthnicity",
        "CensusBlockGroupFipsCode", "MyChartStatus", "SmokingStatus", "VitalStatus"
    ]],
    left_on="PatientDurableKey", right_on="DurableKey",
    how="left"
)

current_year = 2025
journey_pat["ApproxAge"] = current_year - journey_pat["PatientBirthYearBin"]
journey_pat["AgeGroup"]  = pd.cut(
    journey_pat["ApproxAge"],
    bins=[0, 17, 34, 49, 64, 79, 200],
    labels=["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
)

race_map = {
    "White or Caucasian":               "White",
    "Black or African American":        "Black",
    "Hispanic, Latino, or Spanish":     "Hispanic",
    "American Indian or Alaska Native": "Native American",
    "Asian":                            "Asian",
}
journey_pat["RaceSimple"] = journey_pat["FirstRace"].map(race_map).fillna("Other/Unknown")

print("\n--- Long gap rate by age group ---")
print(journey_pat.groupby("AgeGroup", observed=True)["HasLongGap"]
      .agg(["mean", "count"]).round(3).to_string())

print("\n--- Long gap rate by race ---")
print(journey_pat.groupby("RaceSimple")["HasLongGap"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\n--- Long gap rate by MyChart status ---")
print(journey_pat.groupby("MyChartStatus")["HasLongGap"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())


# =============================================================================
# SECTION 5 — SDOH flags per patient
# =============================================================================

print("\nBuilding SDOH flags...")

# Print all unique answer values per domain so we can see what responses exist
print("\n--- Unique AnswerText values per SDOH domain ---")
for domain in soc_det["Domain"].cat.categories:
    vals = soc_det[soc_det["Domain"] == domain]["AnswerText"].value_counts().head(8)
    print(f"\n  [{domain}]")
    print(vals.to_string())

def flag_domain(domain_name, yes_keywords):
    subset = soc_det[soc_det["Domain"] == domain_name].copy()
    mask   = subset["AnswerText"].str.contains(
        "|".join(yes_keywords), case=False, na=False
    )
    return set(subset.loc[mask, "PatientDurableKey"])

transport_pts = flag_domain("Transportation Needs",      ["yes", "always", "often", "sometimes"])
food_pts      = flag_domain("Food insecurity",           ["often true", "sometimes true"])
financial_pts = flag_domain("Financial Resource Strain", ["yes", "hard", "very hard"])
housing_pts   = flag_domain("Housing Stability",         ["yes", "2", "3", "4", "5"])
stress_pts    = flag_domain("stress",                    ["yes"])

print(f"\n  Transport barrier patients : {len(transport_pts):,}")
print(f"  Food insecurity patients   : {len(food_pts):,}")
print(f"  Financial strain patients  : {len(financial_pts):,}")
print(f"  Housing instability pts    : {len(housing_pts):,}")
print(f"  Stress patients            : {len(stress_pts):,}")

journey_pat["TransportBarrier"]   = journey_pat["PatientDurableKey"].isin(transport_pts)
journey_pat["FoodInsecurity"]     = journey_pat["PatientDurableKey"].isin(food_pts)
journey_pat["FinancialStrain"]    = journey_pat["PatientDurableKey"].isin(financial_pts)
journey_pat["HousingInstability"] = journey_pat["PatientDurableKey"].isin(housing_pts)
journey_pat["Stress"]             = journey_pat["PatientDurableKey"].isin(stress_pts)

sdoh_flags = ["TransportBarrier", "FoodInsecurity", "FinancialStrain",
              "HousingInstability", "Stress"]
journey_pat["SDOHRiskCount"] = journey_pat[sdoh_flags].sum(axis=1)
journey_pat["HighRiskSDOH"]  = journey_pat["SDOHRiskCount"] >= 2

print(f"\nPatients with 0 SDOH risks  : {(journey_pat['SDOHRiskCount']==0).sum():,}")
print(f"Patients with 1 SDOH risk   : {(journey_pat['SDOHRiskCount']==1).sum():,}")
print(f"Patients with 2+ SDOH risks : {(journey_pat['SDOHRiskCount']>=2).sum():,}")


# =============================================================================
# SECTION 6 — Core results
# =============================================================================

print("\n\n=== CORE RESULTS ===")

print("\n--- Long Gap Rate (60+ days) by SDOH Factor ---")
for flag in sdoh_flags:
    grp      = journey_pat.groupby(flag)["HasLongGap"].agg(["mean", "count"])
    no_rate  = grp.loc[False, "mean"] * 100 if False in grp.index else 0
    yes_rate = grp.loc[True,  "mean"] * 100 if True  in grp.index else 0
    yes_n    = grp.loc[True,  "count"]       if True  in grp.index else 0
    print(f"  {flag:<22} NO: {no_rate:.1f}%   YES: {yes_rate:.1f}%   "
          f"(n with flag={yes_n:,})")

print("\n--- High Risk SDOH (2+ factors) vs Journey Outcomes ---")
print(journey_pat.groupby("HighRiskSDOH")[
    ["HasLongGap", "HasVeryLongGap", "HasED", "NumVisits", "JourneyDays"]
].mean().round(3).to_string())

print("\n--- Median Max Gap Days by SDOH Risk Count ---")
print(journey_pat.groupby("SDOHRiskCount")["MaxGapDays"]
      .agg(["mean", "median", "count"]).round(1).to_string())

print("\n--- Transport Barrier Rate by Race ---")
print(journey_pat.groupby("RaceSimple")["TransportBarrier"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\n--- Transport Barrier Rate by Age Group ---")
print(journey_pat.groupby("AgeGroup", observed=True)["TransportBarrier"]
      .agg(["mean", "count"]).round(3).to_string())


# =============================================================================
# SECTION 7 — Visualizations
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Fracture Patient Journeys & Social Determinants of Health",
             fontsize=14, fontweight="bold")

# Plot 1: Long gap rate by SDOH flag
gap_rates = {}
for flag in sdoh_flags:
    grp = journey_pat.groupby(flag)["HasLongGap"].mean()
    gap_rates[flag] = {
        "No barrier":  grp.get(False, 0) * 100,
        "Has barrier": grp.get(True,  0) * 100,
    }
gap_df = pd.DataFrame(gap_rates).T
gap_df.plot(kind="bar", ax=axes[0,0], color=["steelblue", "tomato"])
axes[0,0].set_title("60+ Day Gap Rate by SDOH Factor")
axes[0,0].set_ylabel("% of Journeys with Long Gap")
axes[0,0].tick_params(axis="x", rotation=30)
axes[0,0].legend(["No barrier", "Has barrier"])

# Plot 2: SDOH risk count vs median max gap
risk_gap = journey_pat.groupby("SDOHRiskCount")["MaxGapDays"].median()
if not risk_gap.empty:
    risk_gap.plot(kind="bar", ax=axes[0,1], color="steelblue")
axes[0,1].set_title("Median Max Gap Days by # SDOH Risk Factors")
axes[0,1].set_xlabel("# SDOH Risk Factors")
axes[0,1].set_ylabel("Median Max Gap (days)")
axes[0,1].tick_params(axis="x", rotation=0)

# Plot 3: Long gap rate by age group
age_gap = journey_pat.groupby("AgeGroup", observed=True)["HasLongGap"].mean() * 100
if not age_gap.empty:
    age_gap.plot(kind="bar", ax=axes[0,2], color="mediumpurple")
axes[0,2].set_title("60+ Day Gap Rate by Age Group")
axes[0,2].set_ylabel("% Journeys with Long Gap")
axes[0,2].tick_params(axis="x", rotation=30)

# Plot 4: Long gap rate by race
race_gap = (journey_pat.groupby("RaceSimple")["HasLongGap"]
            .mean().sort_values(ascending=True) * 100)
if not race_gap.empty:
    race_gap.plot(kind="barh", ax=axes[1,0], color="coral")
axes[1,0].set_title("60+ Day Gap Rate by Race/Ethnicity")
axes[1,0].set_xlabel("% Journeys with Long Gap")

# Plot 5: Transport barrier rate by race
transport_race = (journey_pat.groupby("RaceSimple")["TransportBarrier"]
                  .mean().sort_values(ascending=True) * 100)
if not transport_race.empty:
    transport_race.plot(kind="barh", ax=axes[1,1], color="teal")
axes[1,1].set_title("Transport Barrier Rate by Race")
axes[1,1].set_xlabel("% with Transport Barrier")

# Plot 6: Journey length by SDOH risk
low  = journey_pat[~journey_pat["HighRiskSDOH"]]["JourneyDays"].clip(upper=500)
high = journey_pat[ journey_pat["HighRiskSDOH"]]["JourneyDays"].clip(upper=500)
if not low.empty:
    axes[1,2].hist(low,  bins=50, alpha=0.6, color="steelblue",
                   label="Low SDOH Risk", density=True)
if not high.empty:
    axes[1,2].hist(high, bins=50, alpha=0.6, color="tomato",
                   label="High SDOH Risk (2+)", density=True)
axes[1,2].set_title("Journey Length: High vs Low SDOH Risk")
axes[1,2].set_xlabel("Journey Length (days, clipped at 500)")
axes[1,2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "journey_analysis.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/journey_analysis.png")


# =============================================================================
# SECTION 8 — Key numbers + save
# =============================================================================

print("\n\n=== KEY NUMBERS FOR PRESENTATION ===")

total_journeys    = len(journey_pat)
long_gap_n        = journey_pat["HasLongGap"].sum()
long_gap_pct      = 100 * journey_pat["HasLongGap"].mean()
transport_long    = journey_pat[ journey_pat["TransportBarrier"]]["HasLongGap"].mean() * 100
no_transport_long = journey_pat[~journey_pat["TransportBarrier"]]["HasLongGap"].mean() * 100
high_risk_ed      = journey_pat[ journey_pat["HighRiskSDOH"]]["HasED"].mean() * 100
low_risk_ed       = journey_pat[~journey_pat["HighRiskSDOH"]]["HasED"].mean() * 100

print(f"\nTotal fracture journeys analyzed : {total_journeys:,}")
print(f"Journeys with 60+ day gap        : {long_gap_n:,}  ({long_gap_pct:.1f}%)")
print(f"\nTransport barrier → long gap     : {transport_long:.1f}%")
print(f"No transport barrier → long gap  : {no_transport_long:.1f}%")
if no_transport_long > 0:
    print(f"Relative increase                : {transport_long/no_transport_long:.2f}x")
print(f"\nHigh SDOH risk → ED visit rate   : {high_risk_ed:.1f}%")
print(f"Low SDOH risk  → ED visit rate   : {low_risk_ed:.1f}%")

journey_pat.to_csv(OUT_DIR / "journey_pat_fractures.csv", index=False)
print("\n✓ Saved outputs/journey_pat_fractures.csv")
print("\n=== DONE — check logs/02_journey_analysis.txt and outputs/ ===")