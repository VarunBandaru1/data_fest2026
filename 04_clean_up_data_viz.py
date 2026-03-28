# =============================================================================
# 04_cleanup_and_viz.py  —  DataFest 2026
# Fixes limitations from 03, produces presentation-ready figures
# Changes:
#   1. Exclude single-visit journeys from gap analysis
#   2. Separate osteoporosis from traumatic fractures
#   3. Add surveyed vs unsurveyed SDOH flag
#   4. Named clusters + publication-quality charts
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

log_file   = open(LOG_DIR / "04_cleanup_and_viz.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)

# Presentation color palette
CLUSTER_COLORS = {
    "Routine Healers":    "#4C9BE8",
    "Lost to Follow-Up":  "#E8944C",
    "Social Crisis":      "#E84C4C",
    "ED Dependent":       "#9B4CE8",
}
SDOH_COLOR    = "#E84C4C"
NO_SDOH_COLOR = "#4C9BE8"

print("✓ Imports OK")


# =============================================================================
# SECTION 1 — Load final journey table + SDOH data
# =============================================================================

print("\nLoading data...")
journey_pat = pd.read_csv(OUT_DIR / "journey_pat_final.csv", low_memory=False)
soc_det = pd.read_csv(
    DATA_DIR / "social_determinants.csv",
    dtype={"EncounterKey": "string", "PatientDurableKey": "string",
           "Domain": "category", "DisplayName": "string", "AnswerText": "string"},
    low_memory=False
)

# Fix key types
journey_pat["PatientDurableKey"] = (
    journey_pat["PatientDurableKey"].astype(str).str.strip().str.replace(".0","",regex=False)
)
soc_det["PatientDurableKey"] = (
    soc_det["PatientDurableKey"].astype(str).str.strip().str.replace(".0","",regex=False)
)

print(f"  journey_pat : {journey_pat.shape}")
print(f"  soc_det     : {soc_det.shape}")


# =============================================================================
# SECTION 2 — Fix 1: Add surveyed vs unsurveyed flag
# =============================================================================

print("\n--- Fix 1: Surveyed vs Unsurveyed ---")

surveyed_patients = set(soc_det["PatientDurableKey"].unique())
journey_pat["WasSurveyed"] = journey_pat["PatientDurableKey"].isin(surveyed_patients)

n_surveyed   = journey_pat["WasSurveyed"].sum()
n_unsurveyed = (~journey_pat["WasSurveyed"]).sum()
print(f"Journeys from surveyed patients   : {n_surveyed:,}  ({100*n_surveyed/len(journey_pat):.1f}%)")
print(f"Journeys from unsurveyed patients : {n_unsurveyed:,}  ({100*n_unsurveyed/len(journey_pat):.1f}%)")

# Among surveyed — how many had no SDOH flags (truly no barrier)
surveyed_only = journey_pat[journey_pat["WasSurveyed"]]
print(f"\nAmong surveyed patients:")
print(f"  With 0 SDOH risks : {(surveyed_only['SDOHRiskCount']==0).sum():,}")
print(f"  With 1+ SDOH risks: {(surveyed_only['SDOHRiskCount']>=1).sum():,}")

# Corrected SDOH comparison — surveyed patients only
print("\n--- ED rate (surveyed patients only, honest comparison) ---")
surv = journey_pat[journey_pat["WasSurveyed"]].copy()
print(surv.groupby("HighRiskSDOH")[["HasED", "HasLongGap", "NumVisits", "JourneyDays"]]
      .mean().round(3).to_string())

print("\n--- ED rate by SDOH factor (surveyed only) ---")
sdoh_flags = ["TransportBarrier", "FoodInsecurity", "FinancialStrain",
              "HousingInstability", "HighStress", "UtilitiesHardship"]
for flag in sdoh_flags:
    if flag not in surv.columns:
        continue
    grp      = surv.groupby(flag)["HasED"].agg(["mean", "count"])
    no_rate  = grp.loc[False, "mean"] * 100 if False in grp.index else 0
    yes_rate = grp.loc[True,  "mean"] * 100 if True  in grp.index else 0
    yes_n    = grp.loc[True,  "count"]       if True  in grp.index else 0
    print(f"  {flag:<22} NO: {no_rate:.1f}%   YES: {yes_rate:.1f}%   (n={yes_n:,})")


# =============================================================================
# SECTION 3 — Fix 2: Separate traumatic fractures from osteoporosis
# =============================================================================

print("\n--- Fix 2: Traumatic vs Osteoporosis separation ---")

osteo_names = ["Osteoporosis with current pathological fracture",
               "Osteoporosis without current pathological fracture"]

journey_pat["IsOsteoporosis"] = journey_pat["GroupName"].isin(osteo_names)
journey_pat["IsTraumatic"]    = ~journey_pat["IsOsteoporosis"]

traumatic = journey_pat[journey_pat["IsTraumatic"]].copy()
osteo     = journey_pat[journey_pat["IsOsteoporosis"]].copy()

print(f"Traumatic fracture journeys  : {len(traumatic):,}")
print(f"Osteoporosis journeys        : {len(osteo):,}")

print("\nTraumatic fracture — gap stats:")
print(traumatic["MaxGapDays"].describe().round(1))
print("\nOsteoporosis — gap stats:")
print(osteo["MaxGapDays"].describe().round(1))

print("\nTraumatic — long gap rate by age group:")
print(traumatic.groupby("AgeGroup", observed=True)["HasLongGap"]
      .agg(["mean","count"]).round(3).to_string())

print("\nOsteoporosis — long gap rate by age group:")
print(osteo.groupby("AgeGroup", observed=True)["HasLongGap"]
      .agg(["mean","count"]).round(3).to_string())


# =============================================================================
# SECTION 4 — Fix 3: Multi-visit journeys only for gap analysis
# =============================================================================

print("\n--- Fix 3: Multi-visit journeys (NumVisits >= 2) ---")

multi = journey_pat[journey_pat["NumVisits"] >= 2].copy()
print(f"Multi-visit journeys         : {len(multi):,}  "
      f"({100*len(multi)/len(journey_pat):.1f}% of total)")
print(f"Single-visit journeys removed: {(journey_pat['NumVisits']==1).sum():,}")

print("\nGap stats (multi-visit only):")
print(multi["MaxGapDays"].describe().round(1))

print("\nLong gap rate by fracture type (multi-visit only):")
print(multi.groupby("GroupName")["HasLongGap"]
      .agg(["mean","count"]).sort_values("mean", ascending=False).round(3).to_string())

print("\nLong gap rate by age group (multi-visit, traumatic only):")
multi_traum = multi[multi["IsTraumatic"]]
print(multi_traum.groupby("AgeGroup", observed=True)["HasLongGap"]
      .agg(["mean","count"]).round(3).to_string())

print("\nED rate by transport barrier (multi-visit, surveyed, traumatic):")
mts = multi[multi["IsTraumatic"] & multi["WasSurveyed"]]
print(f"  With transport barrier    : {mts[mts['TransportBarrier']]['HasED'].mean()*100:.1f}%  "
      f"(n={mts['TransportBarrier'].sum():,})")
print(f"  Without transport barrier : {mts[~mts['TransportBarrier']]['HasED'].mean()*100:.1f}%  "
      f"(n={(~mts['TransportBarrier']).sum():,})")


# =============================================================================
# SECTION 5 — Fix 4: Named clusters
# =============================================================================

print("\n--- Fix 4: Rebuilding clusters on traumatic multi-visit journeys ---")

cluster_input = multi[multi["IsTraumatic"]].copy()

feature_cols = ["NumVisits", "JourneyDays", "MaxGapDays", "AvgGapDays",
                "HasED", "HasHospital", "HasLongGap",
                "TransportBarrier", "FoodInsecurity", "FinancialStrain",
                "SDOHRiskCount", "ApproxAge"]

# Fill missing
for col in feature_cols:
    if col in cluster_input.columns:
        cluster_input[col] = pd.to_numeric(cluster_input[col], errors="coerce").fillna(0)

X = cluster_input[feature_cols].values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_input["ClusterID"] = km.fit_predict(X_scaled)

# Profile clusters to assign names
profile = cluster_input.groupby("ClusterID")[feature_cols].mean()
print("\nCluster profiles:")
print(profile.round(2).to_string())

# Assign names based on profile
# Highest ED → ED Dependent
# Highest MaxGapDays → Lost to Follow-Up
# Highest SDOHRiskCount → Social Crisis
# Remainder → Routine Healers
ed_cluster   = profile["HasED"].idxmax()
gap_cluster  = profile["MaxGapDays"].idxmax()
sdoh_cluster = profile["SDOHRiskCount"].idxmax()
routine_cluster = [i for i in range(4)
                   if i not in [ed_cluster, gap_cluster, sdoh_cluster]][0]

cluster_name_map = {
    ed_cluster:      "ED Dependent",
    gap_cluster:     "Lost to Follow-Up",
    sdoh_cluster:    "Social Crisis",
    routine_cluster: "Routine Healers",
}
# Handle collision if two metrics point to same cluster
seen = {}
for cid, name in cluster_name_map.items():
    if cid not in seen:
        seen[cid] = name
cluster_name_map = seen

cluster_input["ClusterName"] = cluster_input["ClusterID"].map(cluster_name_map)

print("\nCluster sizes with names:")
print(cluster_input["ClusterName"].value_counts().to_string())

print("\nNamed cluster profiles:")
named_profile = cluster_input.groupby("ClusterName")[
    ["NumVisits","JourneyDays","MaxGapDays","HasED","HasLongGap","SDOHRiskCount","ApproxAge"]
].mean().round(2)
print(named_profile.to_string())

print("\nCluster age breakdown:")
print(cluster_input.groupby("ClusterName")["AgeGroup"]
      .value_counts(normalize=True).round(3).to_string())

print("\nCluster race breakdown:")
print(cluster_input.groupby("ClusterName")["RaceSimple"]
      .value_counts(normalize=True).round(3).to_string())


# =============================================================================
# SECTION 6 — Presentation figures
# =============================================================================

print("\n--- Generating presentation figures ---")
plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Figure 1: The two patient pathway stories ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Two Ways Fracture Patients Fall Through the Cracks",
             fontsize=15, fontweight="bold", y=1.02)

# Left: Long gap rate by age group (traumatic, multi-visit)
age_gap = (multi_traum.groupby("AgeGroup", observed=True)["HasLongGap"]
           .mean() * 100).reset_index()
bars = axes[0].bar(age_gap["AgeGroup"].astype(str),
                   age_gap["HasLongGap"],
                   color=["#4C9BE8" if a not in ["65-79","80+"] else "#E84C4C"
                          for a in age_gap["AgeGroup"].astype(str)])
axes[0].set_title("Long Gap (60+ days) Rate\nby Age Group — Traumatic Fractures",
                  fontsize=12, fontweight="bold")
axes[0].set_ylabel("% of Multi-Visit Journeys with 60+ Day Gap")
axes[0].set_xlabel("Age Group")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
for bar, val in zip(bars, age_gap["HasLongGap"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# Right: ED rate by SDOH factor (surveyed patients only)
sdoh_display = {
    "TransportBarrier":   "Transport\nBarrier",
    "FoodInsecurity":     "Food\nInsecurity",
    "FinancialStrain":    "Financial\nStrain",
    "HousingInstability": "Housing\nInstability",
}
ed_no, ed_yes, labels = [], [], []
surv_traum = surv[surv["IsTraumatic"]] if "IsTraumatic" in surv.columns else surv
for flag, label in sdoh_display.items():
    if flag not in surv_traum.columns:
        continue
    grp = surv_traum.groupby(flag)["HasED"].mean()
    ed_no.append(grp.get(False, 0) * 100)
    ed_yes.append(grp.get(True,  0) * 100)
    labels.append(label)

x = np.arange(len(labels))
w = 0.35
b1 = axes[1].bar(x - w/2, ed_no,  w, label="No Barrier",  color=NO_SDOH_COLOR, alpha=0.85)
b2 = axes[1].bar(x + w/2, ed_yes, w, label="Has Barrier", color=SDOH_COLOR,    alpha=0.85)
axes[1].set_title("ED Visit Rate by Social Barrier\n(Surveyed Patients Only)",
                  fontsize=12, fontweight="bold")
axes[1].set_ylabel("% of Journeys with ED Visit")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=10)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
axes[1].legend()
for bar, val in zip(b2, ed_yes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=SDOH_COLOR,
                 fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_two_pathways.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig1_two_pathways.png")


# ── Figure 2: Cluster map ─────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Four Types of Fracture Patient Journeys",
             fontsize=15, fontweight="bold")

# PCA scatter
cluster_order = ["Routine Healers", "Lost to Follow-Up", "Social Crisis", "ED Dependent"]
for name in cluster_order:
    if name not in CLUSTER_COLORS:
        continue
    mask = cluster_input["ClusterName"] == name
    if mask.sum() == 0:
        continue
    axes[0].scatter(
        X_pca[mask.values, 0], X_pca[mask.values, 1],
        c=CLUSTER_COLORS[name], label=f"{name} (n={mask.sum():,})",
        alpha=0.25, s=10
    )
axes[0].set_title(f"Patient Journey Clusters (PCA)\n"
                  f"{pca.explained_variance_ratio_.sum():.1%} variance explained",
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
legend = axes[0].legend(markerscale=4, fontsize=9, framealpha=0.9)

# Cluster profile bar chart
profile_display = named_profile[
    ["NumVisits", "JourneyDays", "MaxGapDays", "HasED", "HasLongGap"]
].copy()
profile_display.columns = ["Avg Visits", "Journey Days", "Max Gap Days",
                            "ED Rate", "Long Gap Rate"]

profile_norm = (profile_display - profile_display.min()) / \
               (profile_display.max() - profile_display.min() + 1e-9)

x     = np.arange(len(profile_display.columns))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, name in enumerate(named_profile.index):
    color = CLUSTER_COLORS.get(name, "gray")
    axes[1].bar(x + offsets[i] * width, profile_norm.loc[name],
                width, label=name, color=color, alpha=0.85)

axes[1].set_title("Cluster Feature Profiles\n(normalized 0–1)",
                  fontsize=12, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(profile_display.columns, fontsize=10)
axes[1].set_ylabel("Normalized Score")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_cluster_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig2_cluster_map.png")


# ── Figure 3: Osteoporosis vs traumatic gap comparison ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Osteoporosis vs Traumatic Fractures: Different Gap Patterns",
             fontsize=14, fontweight="bold")

# Gap distribution comparison
traum_gaps = multi[multi["IsTraumatic"]]["MaxGapDays"].dropna().clip(upper=400)
osteo_gaps = multi[multi["IsOsteoporosis"]]["MaxGapDays"].dropna().clip(upper=400)

axes[0].hist(traum_gaps, bins=40, alpha=0.65, color="#4C9BE8",
             label=f"Traumatic (n={len(traum_gaps):,})", density=True)
axes[0].hist(osteo_gaps, bins=40, alpha=0.65, color="#E8944C",
             label=f"Osteoporosis (n={len(osteo_gaps):,})", density=True)
axes[0].axvline(traum_gaps.median(), color="#4C9BE8", lw=2, ls="--",
                label=f"Traumatic median: {traum_gaps.median():.0f}d")
axes[0].axvline(osteo_gaps.median(), color="#E8944C", lw=2, ls="--",
                label=f"Osteoporosis median: {osteo_gaps.median():.0f}d")
axes[0].set_title("Max Gap Days Distribution\n(multi-visit journeys, clipped at 400)")
axes[0].set_xlabel("Max Gap Days")
axes[0].set_ylabel("Density")
axes[0].legend(fontsize=9)

# Long gap rate by fracture type (top 8)
top_groups = (multi.groupby("GroupName")["HasLongGap"]
              .agg(["mean","count"])
              .query("count >= 100")
              .sort_values("mean", ascending=True)
              .tail(8))

colors = ["#E8944C" if g in osteo_names else "#4C9BE8"
          for g in top_groups.index]
bars = axes[1].barh(
    [g.replace("Fracture of ", "").replace("Osteoporosis ", "Osteoporosis\n").title()
     for g in top_groups.index],
    top_groups["mean"] * 100,
    color=colors
)
axes[1].set_title("Long Gap Rate by Fracture Type\n(top 8, min 100 journeys)")
axes[1].set_xlabel("% Multi-Visit Journeys with 60+ Day Gap")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
for bar, val in zip(bars, top_groups["mean"] * 100):
    axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)

orange_patch = mpatches.Patch(color="#E8944C", label="Osteoporosis")
blue_patch   = mpatches.Patch(color="#4C9BE8", label="Traumatic Fracture")
axes[1].legend(handles=[orange_patch, blue_patch], fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_osteo_vs_traumatic.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig3_osteo_vs_traumatic.png")


# ── Figure 4: Correlation matrix (clean version) ─────────────────────────────
corr_cols = {
    "Num Visits":       pd.to_numeric(journey_pat["NumVisits"], errors="coerce"),
    "Journey Days":     pd.to_numeric(journey_pat["JourneyDays"], errors="coerce"),
    "Max Gap Days":     pd.to_numeric(journey_pat["MaxGapDays"], errors="coerce").fillna(0),
    "Has Long Gap":     pd.to_numeric(journey_pat["HasLongGap"], errors="coerce"),
    "Has ED Visit":     pd.to_numeric(journey_pat["HasED"], errors="coerce"),
    "Has Hospital":     pd.to_numeric(journey_pat["HasHospital"], errors="coerce"),
    "Age":              pd.to_numeric(journey_pat["ApproxAge"], errors="coerce").fillna(0),
    "Transport":        journey_pat["TransportBarrier"].astype(int) if "TransportBarrier" in journey_pat.columns else 0,
    "Food Insecurity":  journey_pat["FoodInsecurity"].astype(int) if "FoodInsecurity" in journey_pat.columns else 0,
    "Financial Strain": journey_pat["FinancialStrain"].astype(int) if "FinancialStrain" in journey_pat.columns else 0,
    "Housing Instab":   journey_pat["HousingInstability"].astype(int) if "HousingInstability" in journey_pat.columns else 0,
    "SDOH Risk Count":  pd.to_numeric(journey_pat["SDOHRiskCount"], errors="coerce").fillna(0),
    "MyChart Active":   (journey_pat["MyChartStatus"] == "Activated").astype(int),
}
corr_df     = pd.DataFrame(corr_cols)
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, annot_kws={"size": 9},
    square=True
)
ax.set_title("Correlation Matrix: Journey Features & Social Determinants",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig4_correlation_matrix.png")


# =============================================================================
# SECTION 7 — Final verified numbers for presentation
# =============================================================================

print("\n\n=== VERIFIED KEY NUMBERS (use these in presentation) ===")

total       = len(journey_pat)
multi_n     = len(multi)
traum_n     = len(traumatic)
osteo_n     = len(osteo)

long_gap_all   = journey_pat["HasLongGap"].mean() * 100
long_gap_multi = multi["HasLongGap"].mean() * 100
long_gap_traum = multi_traum["HasLongGap"].mean() * 100

ed_overall     = journey_pat["HasED"].mean() * 100

traum_gap_med  = multi[multi["IsTraumatic"]]["MaxGapDays"].median()
osteo_gap_med  = multi[multi["IsOsteoporosis"]]["MaxGapDays"].median()

# SDOH ED rates — surveyed traumatic patients only
mts_transport_ed    = mts[mts["TransportBarrier"]]["HasED"].mean() * 100
mts_no_transport_ed = mts[~mts["TransportBarrier"]]["HasED"].mean() * 100
mts_housing_ed      = mts[mts["HousingInstability"]]["HasED"].mean() * 100 if "HousingInstability" in mts.columns else 0
mts_no_housing_ed   = mts[~mts["HousingInstability"]]["HasED"].mean() * 100 if "HousingInstability" in mts.columns else 0

print(f"""
DATA SCOPE
  Total fracture journeys          : {total:,}
  Multi-visit journeys (≥2 visits) : {multi_n:,}  ({100*multi_n/total:.1f}%)
  Traumatic fracture journeys      : {traum_n:,}
  Osteoporosis journeys            : {osteo_n:,}

QUESTION 1 — Do patients experience long gaps?
  Long gap rate (all journeys)     : {long_gap_all:.1f}%
  Long gap rate (multi-visit only) : {long_gap_multi:.1f}%
  Long gap rate (traumatic, multi) : {long_gap_traum:.1f}%
  Traumatic median max gap         : {traum_gap_med:.0f} days
  Osteoporosis median max gap      : {osteo_gap_med:.0f} days
  Overall ED visit rate            : {ed_overall:.1f}%

QUESTION 2 — Does home environment predict gaps/ED?
  Transport barrier → ED rate      : {mts_transport_ed:.1f}%
  No transport barrier → ED rate   : {mts_no_transport_ed:.1f}%
  Transport barrier relative risk  : {mts_transport_ed/mts_no_transport_ed:.2f}x
  Housing instability → ED rate    : {mts_housing_ed:.1f}%
  No housing instability → ED rate : {mts_no_housing_ed:.1f}%
  Housing instability relative risk: {mts_housing_ed/mts_no_housing_ed:.2f}x  (if >0)

CLUSTER FINDINGS
""")
for name in cluster_input["ClusterName"].unique():
    c = cluster_input[cluster_input["ClusterName"] == name]
    print(f"  [{name}]  n={len(c):,}")
    print(f"    Avg visits    : {c['NumVisits'].mean():.1f}")
    print(f"    Journey days  : {c['JourneyDays'].mean():.0f}")
    print(f"    Max gap days  : {c['MaxGapDays'].mean():.0f}")
    print(f"    ED rate       : {c['HasED'].mean()*100:.1f}%")
    print(f"    Long gap rate : {c['HasLongGap'].mean()*100:.1f}%")
    print(f"    SDOH risk avg : {c['SDOHRiskCount'].mean():.2f}")
    print()

# Save enriched cluster table
cluster_input.to_csv(OUT_DIR / "journey_clusters_final.csv", index=False)
print("✓ Saved outputs/journey_clusters_final.csv")
print("\n=== DONE — check logs/04_cleanup_and_viz.txt and outputs/ ===")
print("\nFigures saved:")
print("  outputs/fig1_two_pathways.png")
print("  outputs/fig2_cluster_map.png")
print("  outputs/fig3_osteo_vs_traumatic.png")
print("  outputs/fig4_correlation_matrix.png")