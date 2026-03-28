# =============================================================================
# 03_advanced_analysis.py  —  DataFest 2026
# Builds on 02 outputs. Fixes demographic merge, fixes SDOH flags,
# adds correlation matrix + patient journey clustering.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

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

log_file   = open(LOG_DIR / "03_advanced_analysis.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid", palette="muted")
print("✓ Imports OK")


# =============================================================================
# SECTION 1 — Load saved journey table + patients
# =============================================================================

print("\nLoading journey table from outputs/...")
journey_pat = pd.read_csv(OUT_DIR / "journey_pat_fractures.csv", low_memory=False)
print(f"  journey_pat : {journey_pat.shape}")
print(f"  Columns     : {list(journey_pat.columns)}")

# Also reload patients fresh for the demographic fix
patients = pd.read_csv(DATA_DIR / "patients.csv", low_memory=False)
print(f"  patients    : {patients.shape}")

# Reload social determinants for SDOH fix
sd_dtypes = {
    "EncounterKey":      "string",
    "PatientDurableKey": "string",
    "Domain":            "category",
    "DisplayName":       "string",
    "AnswerText":        "string",
}
soc_det = pd.read_csv(DATA_DIR / "social_determinants.csv",
                      dtype=sd_dtypes, low_memory=False)
print(f"  soc_det     : {soc_det.shape}")


# =============================================================================
# SECTION 2 — Fix demographic merge
# =============================================================================

print("\n--- Fixing demographic merge ---")
print(f"journey_pat PatientDurableKey dtype : {journey_pat['PatientDurableKey'].dtype}")
print(f"patients DurableKey dtype           : {patients['DurableKey'].dtype}")
print(f"Sample journey keys  : {journey_pat['PatientDurableKey'].head(5).tolist()}")
print(f"Sample patient keys  : {patients['DurableKey'].head(5).tolist()}")

# Force both to same string type, strip whitespace and decimals
journey_pat["PatientDurableKey"] = (
    journey_pat["PatientDurableKey"]
    .astype(str).str.strip().str.replace(".0", "", regex=False)
)
patients["DurableKey"] = (
    patients["DurableKey"]
    .astype(str).str.strip().str.replace(".0", "", regex=False)
)

print(f"\nAfter fix — sample journey keys : {journey_pat['PatientDurableKey'].head(5).tolist()}")
print(f"After fix — sample patient keys : {patients['DurableKey'].head(5).tolist()}")

# Check overlap
overlap = set(journey_pat["PatientDurableKey"]) & set(patients["DurableKey"])
print(f"Key overlap                     : {len(overlap):,} / {journey_pat['PatientDurableKey'].nunique():,}")

# Re-merge demographics
demo_cols = ["DurableKey", "PatientBirthYearBin", "FirstRace", "OmbEthnicity",
             "CensusBlockGroupFipsCode", "MyChartStatus", "SmokingStatus", "VitalStatus"]

# Drop old demographic columns if they exist
drop_cols = [c for c in demo_cols[1:] if c in journey_pat.columns]
journey_pat = journey_pat.drop(columns=drop_cols, errors="ignore")
journey_pat = journey_pat.drop(columns=["DurableKey", "ApproxAge", "AgeGroup",
                                         "RaceSimple"], errors="ignore")

journey_pat = journey_pat.merge(
    patients[demo_cols],
    left_on="PatientDurableKey", right_on="DurableKey",
    how="left"
)

# Rebuild age + race
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

print("\n--- Race distribution after fix ---")
print(journey_pat["RaceSimple"].value_counts().to_string())
print("\n--- Age group distribution after fix ---")
print(journey_pat["AgeGroup"].value_counts().sort_index().to_string())
print("\n--- MyChart status distribution ---")
print(journey_pat["MyChartStatus"].value_counts().to_string())


# =============================================================================
# SECTION 3 — Fix SDOH flags with correct answer keywords
# =============================================================================

print("\n--- Rebuilding SDOH flags with correct answer values ---")

def flag_domain(domain_name, yes_keywords):
    subset = soc_det[soc_det["Domain"] == domain_name].copy()
    mask   = subset["AnswerText"].str.contains(
        "|".join(yes_keywords), case=False, na=False
    )
    return set(subset.loc[mask, "PatientDurableKey"].astype(str)
               .str.strip().str.replace(".0", "", regex=False))

# Fix soc_det PatientDurableKey type too
soc_det["PatientDurableKey"] = (
    soc_det["PatientDurableKey"]
    .astype(str).str.strip().str.replace(".0", "", regex=False)
)

# Transport: "Yes" = has barrier
transport_pts = flag_domain("Transportation Needs", ["yes"])

# Food insecurity: "Sometimes true" or "Often true"
food_pts = flag_domain("Food insecurity", ["sometimes true", "often true"])

# Financial strain: "Yes" = can't pay, or Already shut off
financial_pts = flag_domain("Financial Resource Strain", ["yes", "already shut off"])

# Housing instability: "Yes" = homeless/no steady place, or moved 2+ times
housing_pts = flag_domain("Housing Stability", ["yes", r"^2$", r"^3$", r"^4$", r"^5$"])

# Stress: Rather much or Very much
stress_pts = flag_domain("stress", ["rather much", "very much"])

# Utilities hardship: Hard or Very hard
utilities_pts = flag_domain("Utilities", ["^hard$", "very hard"])

print(f"  Transport barrier   : {len(transport_pts):,} patients")
print(f"  Food insecurity     : {len(food_pts):,} patients")
print(f"  Financial strain    : {len(financial_pts):,} patients")
print(f"  Housing instability : {len(housing_pts):,} patients")
print(f"  High stress         : {len(stress_pts):,} patients")
print(f"  Utilities hardship  : {len(utilities_pts):,} patients")

journey_pat["TransportBarrier"]   = journey_pat["PatientDurableKey"].isin(transport_pts)
journey_pat["FoodInsecurity"]     = journey_pat["PatientDurableKey"].isin(food_pts)
journey_pat["FinancialStrain"]    = journey_pat["PatientDurableKey"].isin(financial_pts)
journey_pat["HousingInstability"] = journey_pat["PatientDurableKey"].isin(housing_pts)
journey_pat["HighStress"]         = journey_pat["PatientDurableKey"].isin(stress_pts)
journey_pat["UtilitiesHardship"]  = journey_pat["PatientDurableKey"].isin(utilities_pts)

sdoh_flags = ["TransportBarrier", "FoodInsecurity", "FinancialStrain",
              "HousingInstability", "HighStress", "UtilitiesHardship"]

journey_pat["SDOHRiskCount"] = journey_pat[sdoh_flags].sum(axis=1)
journey_pat["HighRiskSDOH"]  = journey_pat["SDOHRiskCount"] >= 2
journey_pat["AnySDOH"]       = journey_pat["SDOHRiskCount"] >= 1

print(f"\nPatients with 0 SDOH risks  : {(journey_pat['SDOHRiskCount']==0).sum():,}")
print(f"Patients with 1 SDOH risk   : {(journey_pat['SDOHRiskCount']==1).sum():,}")
print(f"Patients with 2+ SDOH risks : {(journey_pat['SDOHRiskCount']>=2).sum():,}")


# =============================================================================
# SECTION 4 — Basic analysis: gaps, demographics, SDOH
# =============================================================================

print("\n\n=== BASIC ANALYSIS ===")

print("\n--- Long gap (60+d) rate by fracture type ---")
print(journey_pat.groupby("GroupName")["HasLongGap"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\n--- Long gap rate by age group ---")
print(journey_pat.groupby("AgeGroup", observed=True)["HasLongGap"]
      .agg(["mean", "count"]).round(3).to_string())

print("\n--- Long gap rate by race ---")
print(journey_pat.groupby("RaceSimple")["HasLongGap"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\n--- Long gap rate by MyChart status ---")
print(journey_pat.groupby("MyChartStatus")["HasLongGap"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\n--- ED visit rate by SDOH risk ---")
print(journey_pat.groupby("HighRiskSDOH")[["HasED", "HasLongGap", "NumVisits", "JourneyDays"]]
      .mean().round(3).to_string())

print("\n--- Long gap rate by each SDOH factor ---")
for flag in sdoh_flags:
    grp      = journey_pat.groupby(flag)["HasLongGap"].agg(["mean", "count"])
    no_rate  = grp.loc[False, "mean"] * 100 if False in grp.index else 0
    yes_rate = grp.loc[True,  "mean"] * 100 if True  in grp.index else 0
    yes_n    = grp.loc[True,  "count"]       if True  in grp.index else 0
    print(f"  {flag:<22} NO: {no_rate:.1f}%   YES: {yes_rate:.1f}%   (n={yes_n:,})")

print("\n--- ED visit rate by each SDOH factor ---")
for flag in sdoh_flags:
    grp      = journey_pat.groupby(flag)["HasED"].agg(["mean", "count"])
    no_rate  = grp.loc[False, "mean"] * 100 if False in grp.index else 0
    yes_rate = grp.loc[True,  "mean"] * 100 if True  in grp.index else 0
    yes_n    = grp.loc[True,  "count"]       if True  in grp.index else 0
    print(f"  {flag:<22} NO: {no_rate:.1f}%   YES: {yes_rate:.1f}%   (n={yes_n:,})")

print("\n--- Median max gap by SDOH risk count ---")
print(journey_pat.groupby("SDOHRiskCount")["MaxGapDays"]
      .agg(["median", "mean", "count"]).round(1).to_string())

print("\n--- Transport barrier rate by race ---")
print(journey_pat.groupby("RaceSimple")["TransportBarrier"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\n--- Transport barrier rate by age group ---")
print(journey_pat.groupby("AgeGroup", observed=True)["TransportBarrier"]
      .agg(["mean", "count"]).round(3).to_string())


# =============================================================================
# SECTION 5 — Advanced: Correlation matrix
# =============================================================================

print("\n\n=== ADVANCED: CORRELATION MATRIX ===")

# Build numeric feature matrix
corr_cols = {
    "NumVisits":        journey_pat["NumVisits"],
    "JourneyDays":      journey_pat["JourneyDays"],
    "MaxGapDays":       journey_pat["MaxGapDays"].fillna(0),
    "AvgGapDays":       journey_pat["AvgGapDays"].fillna(0),
    "HasLongGap":       journey_pat["HasLongGap"].astype(int),
    "HasVeryLongGap":   journey_pat["HasVeryLongGap"].astype(int),
    "HasED":            journey_pat["HasED"].astype(int),
    "HasHospital":      journey_pat["HasHospital"].astype(int),
    "ApproxAge":        journey_pat["ApproxAge"].fillna(0),
    "TransportBarrier": journey_pat["TransportBarrier"].astype(int),
    "FoodInsecurity":   journey_pat["FoodInsecurity"].astype(int),
    "FinancialStrain":  journey_pat["FinancialStrain"].astype(int),
    "HousingInstab":    journey_pat["HousingInstability"].astype(int),
    "HighStress":       journey_pat["HighStress"].astype(int),
    "UtilitiesHard":    journey_pat["UtilitiesHardship"].astype(int),
    "SDOHRiskCount":    journey_pat["SDOHRiskCount"],
    "MyChartActive":    (journey_pat["MyChartStatus"] == "Activated").astype(int),
}

corr_df = pd.DataFrame(corr_cols)
corr_matrix = corr_df.corr()

print("\nTop correlations with MaxGapDays:")
print(corr_matrix["MaxGapDays"].sort_values(ascending=False).round(3).to_string())

print("\nTop correlations with HasED:")
print(corr_matrix["HasED"].sort_values(ascending=False).round(3).to_string())

print("\nTop correlations with HasLongGap:")
print(corr_matrix["HasLongGap"].sort_values(ascending=False).round(3).to_string())

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, annot_kws={"size": 8}
)
ax.set_title("Correlation Matrix: Journey Features & SDOH Factors", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/correlation_matrix.png")


# =============================================================================
# SECTION 6 — Advanced: Patient journey clustering
# =============================================================================

print("\n\n=== ADVANCED: PATIENT JOURNEY CLUSTERING ===")

# Features for clustering — journey shape and SDOH context
cluster_features = [
    "NumVisits", "JourneyDays", "MaxGapDays", "AvgGapDays",
    "HasED", "HasHospital", "HasLongGap",
    "TransportBarrier", "FoodInsecurity", "FinancialStrain",
    "SDOHRiskCount", "ApproxAge"
]

# Build cluster df — only rows with enough data
cluster_df = corr_df[["NumVisits", "JourneyDays", "MaxGapDays", "AvgGapDays",
                        "HasED", "HasHospital", "HasLongGap",
                        "TransportBarrier", "FoodInsecurity", "FinancialStrain",
                        "SDOHRiskCount", "ApproxAge"]].copy()
cluster_df = cluster_df.fillna(0)

print(f"Clustering on {len(cluster_df):,} journeys with {cluster_df.shape[1]} features")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# Find optimal k using inertia (elbow method)
inertias = []
k_range  = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

print("\nInertia by k (elbow method):")
for k, inertia in zip(k_range, inertias):
    print(f"  k={k}: {inertia:,.0f}")

# Fit with k=4 (good balance of interpretability and granularity)
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
journey_pat["Cluster"] = km.fit_predict(X_scaled)

print(f"\nCluster sizes (k={K}):")
print(journey_pat["Cluster"].value_counts().sort_index().to_string())

# Profile each cluster
print(f"\nCluster profiles (mean values):")
profile_cols = ["NumVisits", "JourneyDays", "MaxGapDays", "HasED",
                "HasLongGap", "TransportBarrier", "FoodInsecurity",
                "FinancialStrain", "SDOHRiskCount", "ApproxAge"]
profile = journey_pat.groupby("Cluster")[profile_cols].mean().round(2)
print(profile.to_string())

# Label clusters based on their profile
print("\n--- Cluster race breakdown ---")
print(journey_pat.groupby("Cluster")["RaceSimple"].value_counts(normalize=True)
      .round(3).to_string())

print("\n--- Cluster fracture type breakdown ---")
print(journey_pat.groupby("Cluster")["GroupName"].value_counts(normalize=True)
      .round(3).head(20).to_string())

print("\n--- Cluster age group breakdown ---")
print(journey_pat.groupby("Cluster")["AgeGroup"].value_counts(normalize=True)
      .round(3).to_string())


# =============================================================================
# SECTION 7 — PCA visualization of clusters
# =============================================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA explained variance: {pca.explained_variance_ratio_.round(3)}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Patient Journey Clusters", fontsize=13, fontweight="bold")

colors = ["steelblue", "tomato", "seagreen", "darkorange"]
cluster_labels = {
    0: f"Cluster 0",
    1: f"Cluster 1",
    2: f"Cluster 2",
    3: f"Cluster 3",
}

# PCA scatter
for c in range(K):
    mask = journey_pat["Cluster"] == c
    axes[0].scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[c], label=cluster_labels[c],
        alpha=0.3, s=8
    )
axes[0].set_title("PCA: 2D Journey Cluster Map")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
axes[0].legend(markerscale=3)

# Cluster profile heatmap
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
sns.heatmap(profile_norm.T, annot=profile.T, fmt=".2f",
            cmap="YlOrRd", ax=axes[1], linewidths=0.5,
            annot_kws={"size": 7})
axes[1].set_title("Cluster Feature Profiles (normalized)")
axes[1].set_xlabel("Cluster")

plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_analysis.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/cluster_analysis.png")


# =============================================================================
# SECTION 8 — Elbow plot
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(k_range), inertias, "o-", color="steelblue", lw=2)
ax.set_title("KMeans Elbow Method")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
plt.tight_layout()
plt.savefig(OUT_DIR / "elbow_plot.png", dpi=130, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/elbow_plot.png")


# =============================================================================
# SECTION 9 — Final key numbers
# =============================================================================

print("\n\n=== FINAL KEY NUMBERS FOR PRESENTATION ===")

total    = len(journey_pat)
long_gap = journey_pat["HasLongGap"].mean() * 100
ed_rate  = journey_pat["HasED"].mean() * 100

high_sdoh_ed  = journey_pat[journey_pat["HighRiskSDOH"]]["HasED"].mean() * 100
low_sdoh_ed   = journey_pat[~journey_pat["HighRiskSDOH"]]["HasED"].mean() * 100

transport_ed  = journey_pat[journey_pat["TransportBarrier"]]["HasED"].mean() * 100
no_transport_ed = journey_pat[~journey_pat["TransportBarrier"]]["HasED"].mean() * 100

osteop_gap = journey_pat[journey_pat["GroupName"].str.contains("Osteoporosis", na=False)]["MaxGapDays"].median()

print(f"\nTotal fracture journeys          : {total:,}")
print(f"Overall long gap rate (60+d)     : {long_gap:.1f}%")
print(f"Overall ED visit rate            : {ed_rate:.1f}%")
print(f"\nHigh SDOH risk → ED rate         : {high_sdoh_ed:.1f}%")
print(f"Low SDOH risk  → ED rate         : {low_sdoh_ed:.1f}%")
print(f"Relative ED risk (high vs low)   : {high_sdoh_ed/low_sdoh_ed:.2f}x")
print(f"\nTransport barrier → ED rate      : {transport_ed:.1f}%")
print(f"No transport barrier → ED rate   : {no_transport_ed:.1f}%")
print(f"\nOsteoporosis median follow-up gap: {osteop_gap:.0f} days")

# Save final enriched table
journey_pat.to_csv(OUT_DIR / "journey_pat_final.csv", index=False)
print("\n✓ Saved outputs/journey_pat_final.csv")
print("\n=== DONE — check logs/03_advanced_analysis.txt and outputs/ ===")