# =============================================================================
# 05_deep_dive_and_ml.py  —  DataFest 2026
#
# Deep-dive analyses + ML models answering the two core challenge questions:
#   Q1: "Do patients experience long gaps between diagnosis and future appointments?"
#   Q2: "Does their home environment tell us anything about this?"
#
# Unified narrative:
#   "In SVH's fracture patient population, do social and environmental barriers
#    predict dangerous gaps in follow-up care, and which communities are most at risk?"
#
# Sections:
#   1. Lost to Follow-Up Cluster  — who disappears and why
#   2. Osteoporosis Pathway       — the silent care crisis (194-day median gap)
#   3. ED Substitution Pattern    — barriers reroute care, they don't stop it
#   4. ML Models                  — Logistic Regression + Random Forest,
#                                   ROC curves, odds ratios, feature importance
#
# Prerequisites: Run 04_clean_up_data_viz.py first to generate:
#   outputs/journey_pat_final.csv
#   outputs/journey_clusters_final.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import warnings
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(".")
OUT_DIR  = Path("outputs")
LOG_DIR  = Path("logs")
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Tee logger ───────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_file   = open(LOG_DIR / "05_deep_dive_ml.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)

# ── Colour palette ────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    "Routine Healers":    "#4C9BE8",
    "Lost to Follow-Up":  "#E8944C",
    "Social Crisis":      "#E84C4C",
    "ED Dependent":       "#9B4CE8",
}
BLUE  = "#4C9BE8"
RED   = "#E84C4C"
AMBER = "#E8944C"

print("✓ Imports OK")
print("="*70)


# =============================================================================
# LOAD & PREPARE DATA
# =============================================================================

print("\nLoading data...")

# ── Check prerequisites ───────────────────────────────────────────────────────
journey_final_path   = OUT_DIR / "journey_pat_final.csv"
clusters_final_path  = OUT_DIR / "journey_clusters_final.csv"

if not journey_final_path.exists():
    print("ERROR: outputs/journey_pat_final.csv not found.")
    print("       Please run 04_clean_up_data_viz.py first, then re-run this script.")
    sys.exit(1)

journey_pat = pd.read_csv(journey_final_path, low_memory=False)

# ── Load raw SDOH data ────────────────────────────────────────────────────────
soc_det = pd.read_csv(
    DATA_DIR / "social_determinants.csv",
    dtype={"EncounterKey": "string", "PatientDurableKey": "string",
           "Domain": "category", "DisplayName": "string", "AnswerText": "string"},
    low_memory=False
)

# ── Normalise key types (same fix applied in scripts 03 & 04) ─────────────────
def fix_key(series):
    return series.astype(str).str.strip().str.replace(".0", "", regex=False)

journey_pat["PatientDurableKey"] = fix_key(journey_pat["PatientDurableKey"])
soc_det["PatientDurableKey"]     = fix_key(soc_det["PatientDurableKey"])

print(f"  journey_pat : {journey_pat.shape}")
print(f"  soc_det     : {soc_det.shape}")


# ── Rebuild SDOH flags fresh (avoids column-conflict issues across runs) ───────
def make_flag(domain_str, condition_fn):
    """Return a patient-level boolean flag for one SDOH domain.
    Always returns a DataFrame with columns ['PatientDurableKey', 'flag'].
    """
    sub = soc_det[soc_det["Domain"] == domain_str][
        ["PatientDurableKey", "AnswerText"]
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["PatientDurableKey", "flag"])
    sub["flag"] = sub["AnswerText"].apply(condition_fn)
    return (sub.groupby("PatientDurableKey")["flag"]
               .max()
               .reset_index())          # column stays "flag" — renamed in the loop

sdoh_definitions = {
    "TransportBarrier":  ("Transportation Needs",
                          lambda x: str(x).strip().lower() == "yes"),
    "FoodInsecurity":    ("Food insecurity",
                          lambda x: str(x).strip().lower() in
                                    ["sometimes true", "often true"]),
    "FinancialStrain":   ("Financial Resource Strain",
                          lambda x: str(x).strip().lower() == "yes"),
    "HousingInstability":("Housing Stability",
                          lambda x: str(x).strip() in ["1", "2", "3", "Yes"]),
    "HighStress":        ("stress",
                          lambda x: str(x).strip().lower() in
                                    ["rather much", "very much"]),
    "UtilitiesHardship": ("Utilities",
                          lambda x: str(x).strip().lower() in
                                    ["somewhat hard", "hard", "very hard"]),
    "IPV":               ("intimate partner violance",   # note: typo in source data
                          lambda x: str(x).strip().lower() == "yes"),
}

# Drop any stale SDOH columns that may already be in the CSV, then re-merge
sdoh_cols_to_rebuild = list(sdoh_definitions.keys())
for col in sdoh_cols_to_rebuild:
    if col in journey_pat.columns:
        journey_pat.drop(columns=[col], inplace=True)

for col, (domain, fn) in sdoh_definitions.items():
    flag_df = make_flag(domain, fn)
    if flag_df.empty or "flag" not in flag_df.columns:
        journey_pat[col] = False
        print(f"  WARNING: no data found for domain '{domain}' → {col} set to False")
    else:
        flag_df = flag_df.rename(columns={"flag": col})   # name it what we expect
        journey_pat = journey_pat.merge(flag_df[["PatientDurableKey", col]],
                                        on="PatientDurableKey", how="left")
        journey_pat[col] = journey_pat[col].fillna(False).astype(bool)
        n_true = journey_pat[col].sum()
        print(f"  {col:<22}: {n_true:,} patients flagged")

# Derived aggregates
journey_pat["SDOHRiskCount"] = journey_pat[sdoh_cols_to_rebuild].sum(axis=1)
journey_pat["HighRiskSDOH"]  = journey_pat["SDOHRiskCount"] >= 2
journey_pat["AnySDoH"]       = journey_pat["SDOHRiskCount"] >= 1

# Surveyed / fracture-type flags
surveyed_pats               = set(soc_det["PatientDurableKey"].unique())
journey_pat["WasSurveyed"]  = journey_pat["PatientDurableKey"].isin(surveyed_pats)

osteo_names = [
    "Osteoporosis with current pathological fracture",
    "Osteoporosis without current pathological fracture",
]
journey_pat["IsOsteoporosis"] = journey_pat["GroupName"].isin(osteo_names)
journey_pat["IsTraumatic"]    = ~journey_pat["IsOsteoporosis"]
journey_pat["IsMultiVisit"]   = journey_pat["NumVisits"] >= 2


# ── Attach cluster names ───────────────────────────────────────────────────────
if clusters_final_path.exists():
    clusters_csv = pd.read_csv(clusters_final_path, low_memory=False)
    clusters_csv["PatientDurableKey"] = fix_key(clusters_csv["PatientDurableKey"].astype(str))
    if "ClusterName" in clusters_csv.columns:
        merge_cols = ["PatientDurableKey", "FirstVisit", "ClusterName"]
        journey_pat = journey_pat.merge(
            clusters_csv[merge_cols].drop_duplicates(),
            on=["PatientDurableKey", "FirstVisit"], how="left"
        )
        print(f"  ClusterName attached from journey_clusters_final.csv  "
              f"({journey_pat['ClusterName'].notna().sum():,} rows matched)")
    else:
        print("  WARNING: ClusterName column not in journey_clusters_final.csv — "
              "will re-cluster below")
        journey_pat["ClusterName"] = np.nan
else:
    print("  WARNING: outputs/journey_clusters_final.csv not found — "
          "will re-cluster below")
    journey_pat["ClusterName"] = np.nan


# ── Re-cluster if necessary ────────────────────────────────────────────────────
if journey_pat["ClusterName"].isna().all():
    print("  Re-running KMeans clustering on traumatic multi-visit journeys...")
    cluster_input = journey_pat[
        journey_pat["IsMultiVisit"] & journey_pat["IsTraumatic"]
    ].copy()

    feature_cols = ["NumVisits", "JourneyDays", "MaxGapDays", "AvgGapDays",
                    "HasED", "HasHospital", "HasLongGap",
                    "TransportBarrier", "FoodInsecurity", "FinancialStrain",
                    "SDOHRiskCount", "ApproxAge"]
    for fc in feature_cols:
        cluster_input[fc] = pd.to_numeric(cluster_input[fc], errors="coerce").fillna(0)

    X_c     = cluster_input[feature_cols].values
    X_c_s   = StandardScaler().fit_transform(X_c)
    km      = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_input["ClusterID"] = km.fit_predict(X_c_s)

    profile    = cluster_input.groupby("ClusterID")[feature_cols].mean()
    ed_cl      = profile["HasED"].idxmax()
    gap_cl     = profile["MaxGapDays"].idxmax()
    sdoh_cl    = profile["SDOHRiskCount"].idxmax()
    routine_cl = [i for i in range(4) if i not in [ed_cl, gap_cl, sdoh_cl]][0]
    name_map   = {ed_cl: "ED Dependent", gap_cl: "Lost to Follow-Up",
                  sdoh_cl: "Social Crisis", routine_cl: "Routine Healers"}
    cluster_input["ClusterName"] = cluster_input["ClusterID"].map(name_map)

    journey_pat = journey_pat.merge(
        cluster_input[["PatientDurableKey", "FirstVisit", "ClusterName"]]
                     .drop_duplicates(),
        on=["PatientDurableKey", "FirstVisit"], how="left"
    )
    print("  Re-clustering done.")


# ── Working subsets ────────────────────────────────────────────────────────────
multi      = journey_pat[journey_pat["IsMultiVisit"]].copy()
multi_t    = multi[multi["IsTraumatic"]].copy()
multi_ts   = multi_t[multi_t["WasSurveyed"]].copy()   # traumatic multi-visit surveyed
osteo      = journey_pat[journey_pat["IsOsteoporosis"]].copy()
osteo_multi= osteo[osteo["IsMultiVisit"]].copy()
traum_multi= journey_pat[journey_pat["IsTraumatic"] & journey_pat["IsMultiVisit"]].copy()
clusters   = journey_pat.dropna(subset=["ClusterName"]).copy()

print(f"\nSubset sizes:")
print(f"  All journeys          : {len(journey_pat):,}")
print(f"  Multi-visit           : {len(multi):,}")
print(f"  Traumatic multi-visit : {len(multi_t):,}")
print(f"  Surv. traum. multi    : {len(multi_ts):,}")
print(f"  Osteoporosis          : {len(osteo):,}")
print(f"  Clustered rows        : {len(clusters):,}")


# =============================================================================
# SECTION 1 — LOST TO FOLLOW-UP DEEP DIVE
# =============================================================================

print("\n" + "="*70)
print("SECTION 1 — LOST TO FOLLOW-UP CLUSTER DEEP DIVE")
print("="*70)

ltfu    = clusters[clusters["ClusterName"] == "Lost to Follow-Up"].copy()
routine = clusters[clusters["ClusterName"] == "Routine Healers"].copy()
social  = clusters[clusters["ClusterName"] == "Social Crisis"].copy()
ed_dep  = clusters[clusters["ClusterName"] == "ED Dependent"].copy()

print(f"\nCluster sizes:")
for cname in ["Routine Healers", "Lost to Follow-Up", "ED Dependent", "Social Crisis"]:
    sub = clusters[clusters["ClusterName"] == cname]
    print(f"  {cname:<22}: n={len(sub):,}  "
          f"mean_gap={sub['MaxGapDays'].mean():.0f}d  "
          f"ED={sub['HasED'].mean()*100:.1f}%  "
          f"SDOH_avg={sub['SDOHRiskCount'].mean():.2f}")

print(f"\n--- Lost to Follow-Up deep dive ---")
print(f"  n                   : {len(ltfu):,}")
print(f"  Mean age            : {ltfu['ApproxAge'].mean():.1f}")
print(f"  Mean max gap        : {ltfu['MaxGapDays'].mean():.0f} days")
print(f"  Median max gap      : {ltfu['MaxGapDays'].median():.0f} days")
print(f"  ED visit rate       : {ltfu['HasED'].mean()*100:.1f}%")
print(f"  Avg SDOH risk count : {ltfu['SDOHRiskCount'].mean():.2f}")

print("\nTop fracture types in Lost to Follow-Up:")
print(ltfu["GroupName"].value_counts(normalize=True).head(6).round(3).to_string())

print("\nAge group breakdown — LTFU vs Routine Healers:")
age_order = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
age_compare = pd.DataFrame({
    "Lost to Follow-Up": ltfu["AgeGroup"].value_counts(normalize=True).reindex(age_order, fill_value=0),
    "Routine Healers":   routine["AgeGroup"].value_counts(normalize=True).reindex(age_order, fill_value=0),
}).round(3)
print(age_compare.to_string())

print("\nGap distribution — LTFU vs Routine:")
print(f"  LTFU    — mean={ltfu['MaxGapDays'].mean():.0f}d, "
      f"median={ltfu['MaxGapDays'].median():.0f}d, "
      f"p75={ltfu['MaxGapDays'].quantile(.75):.0f}d")
print(f"  Routine — mean={routine['MaxGapDays'].mean():.0f}d, "
      f"median={routine['MaxGapDays'].median():.0f}d, "
      f"p75={routine['MaxGapDays'].quantile(.75):.0f}d")

# Chi-squared: LTFU vs Routine — are the age distributions significantly different?
ltfu_age_ct    = ltfu["AgeGroup"].value_counts().reindex(age_order, fill_value=0)
routine_age_ct = routine["AgeGroup"].value_counts().reindex(age_order, fill_value=0)
chi2, p_age, dof, _ = stats.chi2_contingency(
    pd.DataFrame({"LTFU": ltfu_age_ct, "Routine": routine_age_ct})
)
print(f"\nChi-squared (age distribution LTFU vs Routine): χ²={chi2:.1f}, p={p_age:.4f}, df={dof}")

# Mann-Whitney U: gap days LTFU vs Routine
u_stat, p_gap = stats.mannwhitneyu(
    ltfu["MaxGapDays"].dropna(), routine["MaxGapDays"].dropna(), alternative="greater"
)
print(f"Mann-Whitney U (LTFU gap > Routine gap): U={u_stat:.0f}, p={p_gap:.2e}")

# --- FIGURE 5: LTFU Deep Dive ---
frac_rename = {
    "Fracture at wrist and hand level":                     "Wrist / Hand",
    "Fracture of femur":                                    "Femur",
    "Fracture of foot and toe, except ankle":               "Foot / Toe",
    "Fracture of forearm":                                  "Forearm",
    "Fracture of lower leg, including ankle":               "Lower Leg",
    "Fracture of lumbar spine and pelvis":                  "Lumbar / Pelvis",
    "Fracture of shoulder and upper arm":                   "Shoulder / Upper Arm",
    "Fracture of rib":                                      "Rib",
    "Fracture of skull and facial bones":                   "Skull / Face",
    "Fracture of cervical vertebra and other parts of neck":"Cervical",
    "Periprosthetic fracture around internal prosthetic joint": "Periprosthetic",
    "Osteoporosis with current pathological fracture":      "Osteo + Fracture",
    "Osteoporosis without current pathological fracture":   "Osteo (no fracture)",
}

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Lost to Follow-Up: Who Disappears After a Fracture?",
             fontsize=14, fontweight="bold")

# Panel A — top fracture types in LTFU
top_fracs = ltfu["GroupName"].value_counts(normalize=True).head(7)
top_fracs.index = [frac_rename.get(x, x) for x in top_fracs.index]
axes[0].barh(top_fracs.index[::-1], top_fracs.values[::-1] * 100,
             color=CLUSTER_COLORS["Lost to Follow-Up"], alpha=0.88)
axes[0].set_xlabel("% of Lost to Follow-Up patients")
axes[0].set_title("Fracture Types in\nLost to Follow-Up", fontweight="bold")
for i, v in enumerate(top_fracs.values[::-1]):
    axes[0].text(v * 100 + 0.3, i, f"{v*100:.1f}%", va="center", fontsize=8.5)
axes[0].set_xlim(0, top_fracs.values.max() * 115)

# Panel B — age distribution: LTFU vs Routine
ltfu_age  = ltfu["AgeGroup"].value_counts(normalize=True).reindex(age_order, fill_value=0)
rout_age  = routine["AgeGroup"].value_counts(normalize=True).reindex(age_order, fill_value=0)
x = np.arange(len(age_order)); w = 0.35
axes[1].bar(x - w/2, ltfu_age.values * 100, w,
            label="Lost to Follow-Up", color=CLUSTER_COLORS["Lost to Follow-Up"], alpha=0.88)
axes[1].bar(x + w/2, rout_age.values * 100, w,
            label="Routine Healers",   color=CLUSTER_COLORS["Routine Healers"], alpha=0.88)
axes[1].set_xticks(x); axes[1].set_xticklabels(age_order, rotation=30, ha="right")
axes[1].set_ylabel("% of cluster")
axes[1].set_title(f"Age Distribution\n(χ²-test p={p_age:.4f})", fontweight="bold")
axes[1].legend(fontsize=8)

# Panel C — gap distribution all 4 clusters (violin / density)
for cname, color in CLUSTER_COLORS.items():
    sub = clusters[clusters["ClusterName"] == cname]["MaxGapDays"].clip(upper=450)
    axes[2].hist(sub, bins=35, alpha=0.45, label=cname, color=color, density=True)
axes[2].axvline(60, color="black", linestyle="--", lw=1.5, alpha=0.7, label="60-day gap")
axes[2].set_xlabel("Max Gap Days (capped at 450)")
axes[2].set_ylabel("Density")
axes[2].set_title("Gap Distribution by Cluster\n(all 4 clusters)", fontweight="bold")
axes[2].legend(fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_ltfu_deep_dive.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/fig5_ltfu_deep_dive.png")


# =============================================================================
# SECTION 2 — OSTEOPOROSIS PATHWAY DEEP DIVE
# =============================================================================

print("\n" + "="*70)
print("SECTION 2 — OSTEOPOROSIS PATHWAY DEEP DIVE")
print("="*70)

osteo_single = osteo[~osteo["IsMultiVisit"]].copy()

print(f"\nOsteoporosis journeys total         : {len(osteo):,}")
print(f"  Zero follow-up (single visit)     : {len(osteo_single):,}  "
      f"({100*len(osteo_single)/len(osteo):.1f}%)")
print(f"  Multi-visit                       : {len(osteo_multi):,}  "
      f"({100*len(osteo_multi)/len(osteo):.1f}%)")

print(f"\nOsteoporosis multi-visit — gap stats:")
print(osteo_multi["MaxGapDays"].describe().round(1).to_string())

print(f"\nOsteoporosis multi-visit — long gap rate by age:")
print(osteo_multi.groupby("AgeGroup")["HasLongGap"]
                  .agg(["mean", "count"]).round(3).to_string())

print(f"\nTraumatic multi-visit — gap stats (for comparison):")
print(traum_multi["MaxGapDays"].describe().round(1).to_string())

gap_diff = osteo_multi["MaxGapDays"].median() - traum_multi["MaxGapDays"].median()
print(f"\nMedian gap difference (osteo − traumatic): {gap_diff:.0f} days")

# Mann-Whitney U: osteo vs traumatic gap
u_stat2, p_osteo = stats.mannwhitneyu(
    osteo_multi["MaxGapDays"].dropna(),
    traum_multi["MaxGapDays"].dropna(),
    alternative="greater"
)
print(f"Mann-Whitney U (osteo gap > traumatic): U={u_stat2:.0f}, p={p_osteo:.2e}")

# Single-visit osteo age breakdown
print(f"\nAge breakdown — single-visit osteoporosis patients (zero follow-up):")
print(osteo_single["AgeGroup"].value_counts().to_string())

# --- FIGURE 6: Osteoporosis Deep Dive ---
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("The Osteoporosis Silent Crisis: A 194-Day Median Care Gap",
             fontsize=14, fontweight="bold")

# Panel A — donut: follow-up rate
sizes   = [len(osteo_single), len(osteo_multi)]
colors_d= ["#FF6B6B", "#4ECDC4"]
wedges, texts, autotexts = axes[0].pie(
    sizes,
    labels=[f"No Follow-Up\n(Single Visit)\nn={len(osteo_single):,}",
            f"Multi-Visit\nn={len(osteo_multi):,}"],
    autopct="%1.1f%%", colors=colors_d,
    startangle=90, wedgeprops=dict(width=0.55),
    textprops={"fontsize": 9}
)
for at in autotexts: at.set_fontweight("bold")
axes[0].set_title("Osteoporosis Patients:\nDid They Return?", fontweight="bold")

# Panel B — gap distribution: osteo vs traumatic (multi-visit)
bins = np.linspace(0, 500, 45)
axes[1].hist(traum_multi["MaxGapDays"].clip(upper=500), bins=bins,
             alpha=0.60, color=BLUE,
             label=f"Traumatic  (median={int(traum_multi['MaxGapDays'].median())}d)",
             density=True)
axes[1].hist(osteo_multi["MaxGapDays"].clip(upper=500), bins=bins,
             alpha=0.60, color=RED,
             label=f"Osteoporosis  (median={int(osteo_multi['MaxGapDays'].median())}d)",
             density=True)
axes[1].axvline(60,  color="black",  linestyle="--", lw=1.5, label="60-day threshold")
axes[1].axvline(osteo_multi["MaxGapDays"].median(),
                color=RED, linestyle=":",  lw=1.5, alpha=0.8)
axes[1].axvline(traum_multi["MaxGapDays"].median(),
                color=BLUE, linestyle=":", lw=1.5, alpha=0.8)
axes[1].set_xlabel("Max Gap Days"); axes[1].set_ylabel("Density")
axes[1].set_title(f"Gap Distribution\n(Mann-Whitney p={p_osteo:.1e})", fontweight="bold")
axes[1].legend(fontsize=8)

# Panel C — long gap rate by age: osteo vs traumatic (multi-visit)
osteo_lr  = osteo_multi.groupby("AgeGroup")["HasLongGap"].mean().reindex(age_order, fill_value=np.nan)
traum_lr  = traum_multi.groupby("AgeGroup")["HasLongGap"].mean().reindex(age_order, fill_value=np.nan)
x = np.arange(len(age_order)); w = 0.35
axes[2].bar(x - w/2, traum_lr.values * 100, w, color=BLUE,  alpha=0.88, label="Traumatic")
axes[2].bar(x + w/2, osteo_lr.values * 100, w, color=RED,   alpha=0.88, label="Osteoporosis")
axes[2].set_xticks(x); axes[2].set_xticklabels(age_order, rotation=30, ha="right")
axes[2].set_ylabel("Long Gap Rate (%)")
axes[2].set_title("Long Gap Rate by Age Group\n(multi-visit only)", fontweight="bold")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_osteo_deep_dive.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/fig6_osteo_deep_dive.png")


# =============================================================================
# SECTION 3 — ED SUBSTITUTION PATTERN
# =============================================================================

print("\n" + "="*70)
print("SECTION 3 — THE ED SUBSTITUTION PATTERN")
print("="*70)
print("Core paradox: SDOH barriers RAISE ED rates but LOWER (or equal) gap rates.")
print("Interpretation: barriers don't prevent care — they redirect it to the ED.\n")

sdoh_flags   = ["TransportBarrier", "FoodInsecurity", "FinancialStrain",
                "HousingInstability", "HighStress", "UtilitiesHardship", "IPV"]
sdoh_labels  = ["Transport\nBarrier", "Food\nInsecurity", "Financial\nStrain",
                "Housing\nInstability", "High\nStress", "Utilities\nHardship", "IPV"]

print(f"{'SDOH Factor':<22}  {'ED NO':>6}  {'ED YES':>7}  {'Δ ED':>6}  "
      f"{'GAP NO':>7}  {'GAP YES':>8}  {'Δ GAP':>6}  {'n(YES)':>7}  {'χ²-p':>10}")
print("-"*90)

ed_no_list  = []; ed_yes_list  = []
gap_no_list = []; gap_yes_list = []
valid_flags = []; valid_labels = []
chi_results = {}

for flag, label in zip(sdoh_flags, sdoh_labels):
    if flag not in multi_ts.columns:
        continue
    g = multi_ts.groupby(flag)[["HasED", "HasLongGap"]].agg(["mean", "count"])
    if True not in multi_ts[flag].values or False not in multi_ts[flag].values:
        continue
    if True not in g.index or False not in g.index:
        continue

    ed_no   = g.loc[False, ("HasED",    "mean")] * 100
    ed_yes  = g.loc[True,  ("HasED",    "mean")] * 100
    gap_no  = g.loc[False, ("HasLongGap","mean")] * 100
    gap_yes = g.loc[True,  ("HasLongGap","mean")] * 100
    n_yes   = int(g.loc[True,  ("HasED", "count")])

    # Chi-squared for ED ~ flag
    ct_ed  = pd.crosstab(multi_ts[flag], multi_ts["HasED"])
    chi2_ed, p_ed, _, _ = stats.chi2_contingency(ct_ed)
    ct_gap = pd.crosstab(multi_ts[flag], multi_ts["HasLongGap"])
    chi2_g, p_gap_f, _, _ = stats.chi2_contingency(ct_gap)

    chi_results[flag] = {"chi2_ed": chi2_ed, "p_ed": p_ed,
                         "chi2_gap": chi2_g, "p_gap": p_gap_f}

    print(f"  {flag:<22}  {ed_no:>5.1f}%  {ed_yes:>6.1f}%  "
          f"{ed_yes-ed_no:>+5.1f}pp  "
          f"{gap_no:>6.1f}%  {gap_yes:>7.1f}%  "
          f"{gap_yes-gap_no:>+5.1f}pp  "
          f"{n_yes:>6,}  χ²-p={p_ed:.4f}")

    ed_no_list.append(ed_no);   ed_yes_list.append(ed_yes)
    gap_no_list.append(gap_no); gap_yes_list.append(gap_yes)
    valid_flags.append(flag);   valid_labels.append(label)

# Journey length comparison
print("\n--- Journey length: Any SDOH vs None (surveyed, traumatic, multi-visit) ---")
print(multi_ts.groupby("AnySDoH")[
    ["JourneyDays", "NumVisits", "HasED", "HasLongGap"]
].mean().round(3).to_string())

# --- FIGURE 7: ED Substitution ---
if not valid_labels:
    print("\n  WARNING: No SDOH flags had True values in multi_ts — fig7 skipped.")
    print("  Check 'SDOH rebuild' output above; all flags should show n > 0.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "The ED Substitution Pattern: Social Barriers Reroute Care, They Don't Stop It\n"
        "(Surveyed Traumatic Multi-Visit Journeys)",
        fontsize=13, fontweight="bold"
    )

    x = np.arange(len(valid_labels)); w = 0.35

    # Panel A — ED rates
    axes[0].bar(x - w/2, ed_no_list,  w, color=BLUE, alpha=0.88, label="No Barrier")
    axes[0].bar(x + w/2, ed_yes_list, w, color=RED,  alpha=0.88, label="Has Barrier")
    axes[0].set_xticks(x); axes[0].set_xticklabels(valid_labels, fontsize=9)
    axes[0].set_ylabel("ED Visit Rate (%)")
    axes[0].set_title("Emergency Dept Visit Rate\n↑ Barriers → ↑ ED Use", fontweight="bold")
    axes[0].legend()
    axes[0].set_ylim(0, max(ed_yes_list + ed_no_list + [1]) * 1.30)
    for i, (no, yes, flag) in enumerate(zip(ed_no_list, ed_yes_list, valid_flags)):
        p_val = chi_results[flag]["p_ed"]
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        axes[0].annotate(
            f"+{yes-no:.1f}pp\n{stars}",
            xy=(i + w/2, yes + 0.4), ha="center", fontsize=7.5,
            color="darkred" if yes > no else "darkgreen", fontweight="bold"
        )

    # Panel B — Gap rates
    axes[1].bar(x - w/2, gap_no_list,  w, color=BLUE, alpha=0.88, label="No Barrier")
    axes[1].bar(x + w/2, gap_yes_list, w, color=RED,  alpha=0.88, label="Has Barrier")
    axes[1].set_xticks(x); axes[1].set_xticklabels(valid_labels, fontsize=9)
    axes[1].set_ylabel("Long Gap Rate (60+ days, %)")
    axes[1].set_title("Follow-Up Gap Rate (60+ days)\n↑ Barriers → Similar or ↓ Gaps",
                      fontweight="bold")
    axes[1].legend()
    axes[1].set_ylim(0, max(gap_no_list + gap_yes_list + [1]) * 1.30)
    for i, (no, yes, flag) in enumerate(zip(gap_no_list, gap_yes_list, valid_flags)):
        p_val = chi_results[flag]["p_gap"]
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        diff  = yes - no
        axes[1].annotate(
            f"{diff:+.1f}pp\n{stars}",
            xy=(i + w/2, max(yes, no) + 0.4), ha="center", fontsize=7.5,
            color="darkred" if diff > 0 else "darkgreen", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig7_ed_substitution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✓ Saved outputs/fig7_ed_substitution.png")


# =============================================================================
# SECTION 4 — ML MODELS
# =============================================================================

print("\n" + "="*70)
print("SECTION 4 — MACHINE LEARNING MODELS")
print("="*70)
print("Predicting (1) dangerous care gaps and (2) ED dependency")
print("Models: Logistic Regression (interpretable odds ratios) + Random Forest")
print("Dataset: multi-visit, traumatic, surveyed journeys\n")

# ── Build feature matrix ──────────────────────────────────────────────────────
model_df = multi_ts.copy()

# Age ordinal
age_map = {"0-17": 0, "18-34": 1, "35-49": 2, "50-64": 3, "65-79": 4, "80+": 5}
model_df["AgeOrdinal"] = model_df["AgeGroup"].map(age_map).fillna(2)

# Race dummies (White is reference)
model_df["RaceBlack"]    = (model_df["RaceSimple"] == "Black").astype(int)
model_df["RaceHispanic"] = (model_df["RaceSimple"] == "Hispanic").astype(int)
model_df["RaceNativeAm"] = (model_df["RaceSimple"] == "Native American").astype(int)
model_df["RaceAsian"]    = (model_df["RaceSimple"] == "Asian").astype(int)

# MyChart active
model_df["MyChartActive"] = (model_df["MyChartStatus"] == "Activated").astype(int)

# Fracture type dummies (top 6 traumatic types; lower leg is reference)
frac_dummies = {
    "FT_Femur":    "Fracture of femur",
    "FT_Shoulder": "Fracture of shoulder and upper arm",
    "FT_Forearm":  "Fracture of forearm",
    "FT_Wrist":    "Fracture at wrist and hand level",
    "FT_Foot":     "Fracture of foot and toe, except ankle",
    "FT_Lumbar":   "Fracture of lumbar spine and pelvis",
}
for col, gname in frac_dummies.items():
    model_df[col] = (model_df["GroupName"] == gname).astype(int)

# Final feature list
sdoh_feats = ["TransportBarrier", "FoodInsecurity", "FinancialStrain",
              "HousingInstability", "HighStress", "UtilitiesHardship", "IPV"]
demo_feats = ["AgeOrdinal", "RaceBlack", "RaceHispanic", "RaceNativeAm",
              "RaceAsian", "MyChartActive"]
frac_feats = list(frac_dummies.keys())
all_feats  = sdoh_feats + demo_feats + frac_feats

# Convert bool SDOH to int
for col in sdoh_feats:
    model_df[col] = model_df[col].astype(int)

model_df = model_df.dropna(subset=all_feats + ["HasLongGap", "HasED"])
print(f"Modeling dataset: n={len(model_df):,}")
print(f"  HasLongGap prevalence : {model_df['HasLongGap'].mean()*100:.1f}%")
print(f"  HasED prevalence      : {model_df['HasED'].mean()*100:.1f}%")
print(f"  Features              : {len(all_feats)}")

# ── Pretty feature labels ─────────────────────────────────────────────────────
label_map = {
    "TransportBarrier":  "Transport Barrier",
    "FoodInsecurity":    "Food Insecurity",
    "FinancialStrain":   "Financial Strain",
    "HousingInstability":"Housing Instability",
    "HighStress":        "High Stress",
    "UtilitiesHardship": "Utilities Hardship",
    "IPV":               "Intimate Partner Violence",
    "AgeOrdinal":        "Age (older → higher)",
    "RaceBlack":         "Race: Black",
    "RaceHispanic":      "Race: Hispanic",
    "RaceNativeAm":      "Race: Native American",
    "RaceAsian":         "Race: Asian",
    "MyChartActive":     "MyChart Active",
    "FT_Femur":          "Fracture: Femur",
    "FT_Shoulder":       "Fracture: Shoulder",
    "FT_Forearm":        "Fracture: Forearm",
    "FT_Wrist":          "Fracture: Wrist",
    "FT_Foot":           "Fracture: Foot",
    "FT_Lumbar":         "Fracture: Lumbar",
}
feat_labels = [label_map.get(f, f) for f in all_feats]


def run_models(X, y, target_name, feature_names, feat_display, rng=42):
    """Fit LR + RF, cross-validate, return results dict."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=rng, stratify=y
    )
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                            random_state=rng)
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    lr_auc  = roc_auc_score(y_te, lr_prob)

    lr_cv = cross_val_score(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                            C=1.0, random_state=rng))]),
        X, y, cv=cv, scoring="roc_auc"
    )

    # Odds ratios + p-values via statsmodels-style Wald test
    coefs = lr.coef_[0]
    OR    = np.exp(coefs)
    odds_df = pd.DataFrame({
        "feature":     feature_names,
        "display":     feat_display,
        "coefficient": coefs,
        "odds_ratio":  OR,
        "abs_coef":    np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=300, max_depth=7,
                                class_weight="balanced", random_state=rng,
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_auc  = roc_auc_score(y_te, rf_prob)

    rf_cv = cross_val_score(
        RandomForestClassifier(n_estimators=300, max_depth=7,
                               class_weight="balanced", random_state=rng,
                               n_jobs=-1),
        X, y, cv=cv, scoring="roc_auc"
    )

    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "display":    feat_display,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    lr_fpr, lr_tpr, _ = roc_curve(y_te, lr_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_te, rf_prob)

    # Print results
    print(f"\n  ── {target_name} ──")
    print(f"    Logistic Regression  : AUC={lr_auc:.3f}  "
          f"(5-fold CV: {lr_cv.mean():.3f} ± {lr_cv.std():.3f})")
    print(f"    Random Forest        : AUC={rf_auc:.3f}  "
          f"(5-fold CV: {rf_cv.mean():.3f} ± {rf_cv.std():.3f})")
    print(f"\n    Top predictors (Logistic Regression — Odds Ratios):")
    print(odds_df[["display", "odds_ratio", "coefficient"]]
          .head(10).round(3).to_string(index=False))
    print(f"\n    Top predictors (Random Forest — Importance):")
    print(fi_df[["display", "importance"]].head(10).round(4).to_string(index=False))

    return dict(lr=lr, rf=rf, scaler=scaler,
                lr_auc=lr_auc, rf_auc=rf_auc,
                lr_cv=lr_cv, rf_cv=rf_cv,
                odds_df=odds_df, fi_df=fi_df,
                lr_fpr=lr_fpr, lr_tpr=lr_tpr,
                rf_fpr=rf_fpr, rf_tpr=rf_tpr,
                y_te=y_te, feat_display=feat_display)


X = model_df[all_feats].values.astype(float)
y_gap = model_df["HasLongGap"].astype(int).values
y_ed  = model_df["HasED"].astype(int).values

res_gap = run_models(X, y_gap, "HasLongGap — 60+ day care gap", all_feats, feat_labels)
res_ed  = run_models(X, y_ed,  "HasED — Emergency Dept visit",  all_feats, feat_labels)


# ── FIGURE 8: Full ML Results Dashboard ──────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)
fig.suptitle(
    "Predicting Care Gaps & ED Dependency in SVH Fracture Patients\n"
    "Logistic Regression + Random Forest  |  Multi-Visit Traumatic Surveyed Journeys",
    fontsize=13, fontweight="bold", y=1.01
)

# ── ROC Curves ─────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, 0])
ax_roc.plot(res_gap["lr_fpr"], res_gap["lr_tpr"], color=BLUE,  lw=2,
            label=f"LR  — Long Gap   (AUC={res_gap['lr_auc']:.2f})")
ax_roc.plot(res_gap["rf_fpr"], res_gap["rf_tpr"], color=BLUE,  lw=2, ls="--",
            label=f"RF  — Long Gap   (AUC={res_gap['rf_auc']:.2f})")
ax_roc.plot(res_ed["lr_fpr"],  res_ed["lr_tpr"],  color=RED,   lw=2,
            label=f"LR  — ED Visit   (AUC={res_ed['lr_auc']:.2f})")
ax_roc.plot(res_ed["rf_fpr"],  res_ed["rf_tpr"],  color=RED,   lw=2, ls="--",
            label=f"RF  — ED Visit   (AUC={res_ed['rf_auc']:.2f})")
ax_roc.plot([0,1],[0,1], "k:", lw=1.2, alpha=0.4, label="Random (AUC=0.50)")
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves", fontweight="bold")
ax_roc.legend(fontsize=7.5, loc="lower right")
ax_roc.grid(alpha=0.25)

# ── 5-Fold CV AUC Bars ──────────────────────────────────────────────────────
ax_cv = fig.add_subplot(gs[0, 1])
cv_labels = ["LR\nLong Gap", "RF\nLong Gap", "LR\nED Visit", "RF\nED Visit"]
cv_means  = [res_gap["lr_cv"].mean(), res_gap["rf_cv"].mean(),
             res_ed["lr_cv"].mean(),  res_ed["rf_cv"].mean()]
cv_stds   = [res_gap["lr_cv"].std(),  res_gap["rf_cv"].std(),
             res_ed["lr_cv"].std(),   res_ed["rf_cv"].std()]
cv_colors = ["#4C9BE8", "#2E6DA4", "#E84C4C", "#A4200F"]
bars = ax_cv.bar(cv_labels, cv_means, color=cv_colors, alpha=0.88,
                 yerr=cv_stds, capsize=6, error_kw={"elinewidth": 1.8})
ax_cv.axhline(0.5, color="gray", ls=":", lw=1.2, label="Chance (0.50)")
ax_cv.set_ylim(0.35, 1.0); ax_cv.set_ylabel("AUC")
ax_cv.set_title("5-Fold Cross-Validated AUC\n(± 1 std)", fontweight="bold")
ax_cv.legend(fontsize=8)
for bar, m in zip(bars, cv_means):
    ax_cv.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + max(cv_stds)*1.2,
               f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

# ── Odds Ratios — Long Gap (top-right) ──────────────────────────────────────
ax_or_gap = fig.add_subplot(gs[0, 2])
or_gap_top = res_gap["odds_df"].head(12).sort_values("odds_ratio")
c_or_gap   = [RED if v > 1 else BLUE for v in or_gap_top["odds_ratio"]]
ax_or_gap.barh(or_gap_top["display"], or_gap_top["odds_ratio"], color=c_or_gap, alpha=0.85)
ax_or_gap.axvline(1.0, color="black", ls="--", lw=1.2)
ax_or_gap.set_xlabel("Odds Ratio  (> 1 = higher risk)")
ax_or_gap.set_title("LR Odds Ratios\n(Predicting 60+ Day Gap)", fontweight="bold")
ax_or_gap.tick_params(axis="y", labelsize=8)
for i, v in enumerate(or_gap_top["odds_ratio"]):
    ha = "left" if v >= 1 else "right"
    offset = 0.03 if v >= 1 else -0.03
    ax_or_gap.text(v + offset, i, f"{v:.2f}", va="center", ha=ha, fontsize=7.5)

# ── RF Feature Importance — Long Gap (bottom-left) ──────────────────────────
ax_fi_gap = fig.add_subplot(gs[1, 0])
fi_gap_top = res_gap["fi_df"].head(10).sort_values("importance")
ax_fi_gap.barh(fi_gap_top["display"], fi_gap_top["importance"], color=BLUE, alpha=0.85)
ax_fi_gap.set_xlabel("Mean Decrease Impurity")
ax_fi_gap.set_title("RF Feature Importance\n(Predicting Long Gap)", fontweight="bold")
ax_fi_gap.tick_params(axis="y", labelsize=8)

# ── Odds Ratios — ED Visit (bottom-middle) ───────────────────────────────────
ax_or_ed = fig.add_subplot(gs[1, 1])
or_ed_top = res_ed["odds_df"].head(12).sort_values("odds_ratio")
c_or_ed   = [RED if v > 1 else BLUE for v in or_ed_top["odds_ratio"]]
ax_or_ed.barh(or_ed_top["display"], or_ed_top["odds_ratio"], color=c_or_ed, alpha=0.85)
ax_or_ed.axvline(1.0, color="black", ls="--", lw=1.2)
ax_or_ed.set_xlabel("Odds Ratio  (> 1 = higher risk)")
ax_or_ed.set_title("LR Odds Ratios\n(Predicting ED Visit)", fontweight="bold")
ax_or_ed.tick_params(axis="y", labelsize=8)
for i, v in enumerate(or_ed_top["odds_ratio"]):
    ha = "left" if v >= 1 else "right"
    offset = 0.03 if v >= 1 else -0.03
    ax_or_ed.text(v + offset, i, f"{v:.2f}", va="center", ha=ha, fontsize=7.5)

# ── RF Feature Importance — ED Visit (bottom-right) ─────────────────────────
ax_fi_ed = fig.add_subplot(gs[1, 2])
fi_ed_top = res_ed["fi_df"].head(10).sort_values("importance")
ax_fi_ed.barh(fi_ed_top["display"], fi_ed_top["importance"], color=RED, alpha=0.85)
ax_fi_ed.set_xlabel("Mean Decrease Impurity")
ax_fi_ed.set_title("RF Feature Importance\n(Predicting ED Visit)", fontweight="bold")
ax_fi_ed.tick_params(axis="y", labelsize=8)

plt.savefig(OUT_DIR / "fig8_ml_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/fig8_ml_results.png")


# ── FIGURE 9: SDOH Odds Ratios — Presenter-Friendly Summary ─────────────────
# Pull only the SDOH features for a clean "what drives risk" slide
sdoh_only_gap = (res_gap["odds_df"]
                 [res_gap["odds_df"]["feature"].isin(sdoh_feats)]
                 .sort_values("odds_ratio", ascending=True))
sdoh_only_ed  = (res_ed["odds_df"]
                 [res_ed["odds_df"]["feature"].isin(sdoh_feats)]
                 .sort_values("odds_ratio", ascending=True))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Social Determinant Odds Ratios\n"
    "How Much Do Social Barriers Increase Risk? (Logistic Regression, Adjusted)",
    fontsize=13, fontweight="bold"
)

for ax, sdoh_or, title, base_color in [
    (axes[0], sdoh_only_gap, "Predicting 60+ Day Care Gap", BLUE),
    (axes[1], sdoh_only_ed,  "Predicting ED Visit",         RED),
]:
    colors = [RED if v > 1 else BLUE for v in sdoh_or["odds_ratio"]]
    bars = ax.barh(sdoh_or["display"], sdoh_or["odds_ratio"],
                   color=colors, alpha=0.88)
    ax.axvline(1.0, color="black", ls="--", lw=1.5, label="OR = 1 (no effect)")
    ax.set_xlabel("Odds Ratio  (adjusted for age, race, fracture type)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    for i, v in enumerate(sdoh_or["odds_ratio"]):
        ha = "left" if v >= 1 else "right"
        offset = 0.02 if v >= 1 else -0.02
        ax.text(v + offset, i, f"{v:.2f}", va="center", ha=ha,
                fontsize=9, fontweight="bold")
    # Shade risk zone
    ax.axvspan(max(sdoh_or["odds_ratio"]), ax.get_xlim()[1] if ax.get_xlim()[1] > max(sdoh_or["odds_ratio"]) else max(sdoh_or["odds_ratio"])*1.3,
               alpha=0, color=RED)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig9_sdoh_odds_ratios.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig9_sdoh_odds_ratios.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY — KEY NUMBERS FOR PRESENTATION")
print("="*70)

# Safe lookup helpers
def cluster_stat(df, cname, col, fn):
    sub = df[df["ClusterName"] == cname][col]
    return fn(sub) if len(sub) > 0 else float("nan")

def sdoh_ed(flag):
    if flag not in multi_ts.columns or multi_ts[flag].sum() == 0:
        return float("nan"), float("nan")
    return (multi_ts[~multi_ts[flag]]["HasED"].mean() * 100,
            multi_ts[multi_ts[flag]]["HasED"].mean() * 100)

tb_no, tb_yes = sdoh_ed("TransportBarrier")
hi_no, hi_yes = sdoh_ed("HousingInstability")
fs_no, fs_yes = sdoh_ed("FinancialStrain")

print(f"""
SECTION 1 — LOST TO FOLLOW-UP
  Cluster n                     : {len(ltfu):,}
  Median max gap                 : {cluster_stat(clusters, 'Lost to Follow-Up', 'MaxGapDays', np.median):.0f} days
  Long gap rate                  : {cluster_stat(clusters, 'Lost to Follow-Up', 'HasLongGap', np.mean)*100:.1f}%
  ED rate                        : {cluster_stat(clusters, 'Lost to Follow-Up', 'HasED', np.mean)*100:.1f}%
  Avg SDOH risks                 : {cluster_stat(clusters, 'Lost to Follow-Up', 'SDOHRiskCount', np.mean):.2f}
  Top fracture type              : {ltfu['GroupName'].value_counts().index[0] if len(ltfu) > 0 else 'N/A'}
  Chi²-p (age dist. vs Routine)  : {p_age:.4f}
  Mann-Whitney p (gap > Routine) : {p_gap:.2e}

SECTION 2 — OSTEOPOROSIS
  Total journeys                 : {len(osteo):,}
  Zero-follow-up (single visit)  : {len(osteo_single):,}  ({100*len(osteo_single)/len(osteo):.1f}%)
  Median gap (multi-visit)       : {osteo_multi['MaxGapDays'].median():.0f} days
  vs Traumatic median gap        : {traum_multi['MaxGapDays'].median():.0f} days
  Mann-Whitney p (osteo > traum) : {p_osteo:.2e}
  Long gap rate (multi-visit)    : {osteo_multi['HasLongGap'].mean()*100:.1f}%

SECTION 3 — ED SUBSTITUTION
  Transport barrier → ED rate    : {tb_yes:.1f}%  vs  {tb_no:.1f}% (no barrier)
  Housing instability → ED rate  : {hi_yes:.1f}%  vs  {hi_no:.1f}% (no instability)
  Financial strain → ED rate     : {fs_yes:.1f}%  vs  {fs_no:.1f}% (no strain)

SECTION 4 — ML MODELS  (n={len(model_df):,} journeys)
  Long Gap  —  LR AUC  : {res_gap['lr_auc']:.3f}   CV: {res_gap['lr_cv'].mean():.3f} ± {res_gap['lr_cv'].std():.3f}
  Long Gap  —  RF AUC  : {res_gap['rf_auc']:.3f}   CV: {res_gap['rf_cv'].mean():.3f} ± {res_gap['rf_cv'].std():.3f}
  ED Visit  —  LR AUC  : {res_ed['lr_auc']:.3f}   CV: {res_ed['lr_cv'].mean():.3f} ± {res_ed['lr_cv'].std():.3f}
  ED Visit  —  RF AUC  : {res_ed['rf_auc']:.3f}   CV: {res_ed['rf_cv'].mean():.3f} ± {res_ed['rf_cv'].std():.3f}
  Top SDOH predictor (gap)       : {res_gap['odds_df'][res_gap['odds_df']['feature'].isin(sdoh_feats)].sort_values('abs_coef', ascending=False).iloc[0]['display'] if len(res_gap['odds_df']) > 0 else 'N/A'}   OR={res_gap['odds_df'][res_gap['odds_df']['feature'].isin(sdoh_feats)].sort_values('abs_coef', ascending=False).iloc[0]['odds_ratio']:.2f}
  Top SDOH predictor (ED)        : {res_ed['odds_df'][res_ed['odds_df']['feature'].isin(sdoh_feats)].sort_values('abs_coef', ascending=False).iloc[0]['display'] if len(res_ed['odds_df']) > 0 else 'N/A'}   OR={res_ed['odds_df'][res_ed['odds_df']['feature'].isin(sdoh_feats)].sort_values('abs_coef', ascending=False).iloc[0]['odds_ratio']:.2f}

OUTPUT FIGURES
  fig5_ltfu_deep_dive.png
  fig6_osteo_deep_dive.png
  fig7_ed_substitution.png
  fig8_ml_results.png         (full ML dashboard)
  fig9_sdoh_odds_ratios.png   (clean presenter slide)
""")

print("=== DONE — check logs/05_deep_dive_ml.txt and outputs/ ===")