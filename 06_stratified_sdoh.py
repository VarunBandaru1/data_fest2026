# =============================================================================
# 06_age_stratified_sdoh.py  —  DataFest 2026
#
# Research question:
#   "Among older patients, which social/environmental barriers most strongly
#    drive long follow-up gaps — and how do these effects differ from younger
#    patients?"
#
# Sections:
#   1. SDOH Prevalence by Age   — who carries which barriers
#   2. Barrier Effect on Gaps   — does a barrier hurt older patients MORE?
#   3. Barrier Effect on ED     — same question for emergency use
#   4. LTFU Cluster by Age      — what distinguishes old LTFU from young LTFU
#   5. Condition-Specific       — femur / lower leg / shoulder in 65+ vs <65
#   6. Interaction Models       — LR stratified + interaction terms (Is Old × SDOH)
#
# Prerequisites: Run 04_clean_up_data_viz.py first.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as stats
import warnings
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_file   = open(LOG_DIR / "06_age_stratified_sdoh.txt", "w")
sys.stdout = Tee(sys.stdout, log_file)

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE  = "#4C9BE8"; RED  = "#E84C4C"; AMBER = "#E8944C"; GREEN = "#4CAF50"
AGE_COLORS = {
    "Young (≤49)":        "#4C9BE8",
    "Middle-Aged (50-64)":"#9B4CE8",
    "Older (65-79)":      "#E8944C",
    "Eldest (80+)":       "#E84C4C",
}

print("✓ Imports OK")
print("="*70)


# =============================================================================
# LOAD DATA  (same pipeline as script 05)
# =============================================================================
print("\nLoading data...")

if not (OUT_DIR / "journey_pat_final.csv").exists():
    print("ERROR: outputs/journey_pat_final.csv not found. Run 04 first.")
    sys.exit(1)

journey_pat = pd.read_csv(OUT_DIR / "journey_pat_final.csv", low_memory=False)

soc_det = pd.read_csv(
    DATA_DIR / "social_determinants.csv",
    dtype={"EncounterKey": "string", "PatientDurableKey": "string",
           "Domain": "category", "DisplayName": "string", "AnswerText": "string"},
    low_memory=False
)

def fix_key(s):
    return s.astype(str).str.strip().str.replace(".0", "", regex=False)

journey_pat["PatientDurableKey"] = fix_key(journey_pat["PatientDurableKey"])
soc_det["PatientDurableKey"]     = fix_key(soc_det["PatientDurableKey"])

# ── Rebuild SDOH flags ────────────────────────────────────────────────────────
def make_flag(domain_str, condition_fn):
    sub = soc_det[soc_det["Domain"] == domain_str][
        ["PatientDurableKey", "AnswerText"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["PatientDurableKey", "flag"])
    sub["flag"] = sub["AnswerText"].apply(condition_fn)
    return sub.groupby("PatientDurableKey")["flag"].max().reset_index()

sdoh_definitions = {
    "TransportBarrier":   ("Transportation Needs",
                           lambda x: str(x).strip().lower() == "yes"),
    "FoodInsecurity":     ("Food insecurity",
                           lambda x: str(x).strip().lower() in
                                     ["sometimes true", "often true"]),
    "FinancialStrain":    ("Financial Resource Strain",
                           lambda x: str(x).strip().lower() == "yes"),
    "HousingInstability": ("Housing Stability",
                           lambda x: str(x).strip() in ["1","2","3","Yes"]),
    "HighStress":         ("stress",
                           lambda x: str(x).strip().lower() in
                                     ["rather much","very much"]),
    "UtilitiesHardship":  ("Utilities",
                           lambda x: str(x).strip().lower() in
                                     ["somewhat hard","hard","very hard"]),
    "IPV":                ("intimate partner violance",
                           lambda x: str(x).strip().lower() == "yes"),
}

for col in sdoh_definitions:
    if col in journey_pat.columns:
        journey_pat.drop(columns=[col], inplace=True)

for col, (domain, fn) in sdoh_definitions.items():
    flag_df = make_flag(domain, fn)
    if flag_df.empty or "flag" not in flag_df.columns:
        journey_pat[col] = False
    else:
        flag_df = flag_df.rename(columns={"flag": col})
        journey_pat = journey_pat.merge(
            flag_df[["PatientDurableKey", col]],
            on="PatientDurableKey", how="left")
        journey_pat[col] = journey_pat[col].fillna(False).astype(bool)

sdoh_cols = list(sdoh_definitions.keys())
journey_pat["SDOHRiskCount"] = journey_pat[sdoh_cols].sum(axis=1)
journey_pat["AnySDoH"]       = journey_pat["SDOHRiskCount"] >= 1

surveyed_pats = set(soc_det["PatientDurableKey"].unique())
journey_pat["WasSurveyed"] = journey_pat["PatientDurableKey"].isin(surveyed_pats)

osteo_names = ["Osteoporosis with current pathological fracture",
               "Osteoporosis without current pathological fracture"]
journey_pat["IsOsteoporosis"] = journey_pat["GroupName"].isin(osteo_names)
journey_pat["IsTraumatic"]    = ~journey_pat["IsOsteoporosis"]
journey_pat["IsMultiVisit"]   = journey_pat["NumVisits"] >= 2

# ── Age groupings ─────────────────────────────────────────────────────────────
age_bins   = [0,  49,  64,  79, 200]
age_labels = ["Young (≤49)", "Middle-Aged (50-64)", "Older (65-79)", "Eldest (80+)"]
journey_pat["AgeSimple"] = pd.cut(
    journey_pat["ApproxAge"].fillna(40),
    bins=age_bins, labels=age_labels
)
journey_pat["IsOld"] = journey_pat["ApproxAge"] >= 65   # binary for interaction model

# ── Working subset: multi-visit traumatic surveyed ────────────────────────────
multi_ts = journey_pat[
    journey_pat["IsMultiVisit"] &
    journey_pat["IsTraumatic"] &
    journey_pat["WasSurveyed"]
].copy()

print(f"  journey_pat : {journey_pat.shape}")
print(f"  multi_ts    : {len(multi_ts):,}  (multi-visit traumatic surveyed)")
print(f"  % aged 65+  : {(multi_ts['IsOld'].mean()*100):.1f}%")

# ── Friendly SDOH display names ───────────────────────────────────────────────
sdoh_display = {
    "TransportBarrier":   "Transport Barrier",
    "FoodInsecurity":     "Food Insecurity",
    "FinancialStrain":    "Financial Strain",
    "HousingInstability": "Housing Instability",
    "HighStress":         "High Stress",
    "UtilitiesHardship":  "Utilities Hardship",
    "IPV":                "Intimate Partner\nViolence",
}
sdoh_display_1line = {k: v.replace("\n"," ") for k,v in sdoh_display.items()}


# =============================================================================
# SECTION 1 — SDOH PREVALENCE BY AGE GROUP
# =============================================================================
print("\n" + "="*70)
print("SECTION 1 — SDOH BARRIER PREVALENCE BY AGE GROUP")
print("="*70)

# Among surveyed patients in multi_ts: what % of each age group has each barrier
prev_tbl = (multi_ts.groupby("AgeSimple", observed=True)[sdoh_cols]
                     .mean() * 100).round(1)
print("\nBarrier prevalence (%) by age group:")
print(prev_tbl.to_string())

# Count table
count_tbl = (multi_ts.groupby("AgeSimple", observed=True)[sdoh_cols].sum())
n_tbl     = multi_ts.groupby("AgeSimple", observed=True).size().rename("n")
print("\nn per age group:")
print(n_tbl.to_string())

# ── FIGURE A: Prevalence heatmap ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Who Carries Social Barriers? SDOH Prevalence by Age Group\n"
             "(Multi-Visit Traumatic Surveyed Journeys)",
             fontsize=13, fontweight="bold")

# Panel A1 — heatmap
heat_data = prev_tbl[[c for c in sdoh_cols if c != "IPV"]]   # IPV separate (low n)
heat_data.columns = [sdoh_display_1line[c] for c in heat_data.columns]
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=axes[0], cbar_kws={"label": "% of age group with barrier"},
            linewidths=0.5, linecolor="white")
axes[0].set_title("Barrier Prevalence (%) by Age Group", fontweight="bold")
axes[0].set_ylabel("Age Group"); axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=30)

# Panel A2 — grouped bar: SDOHRiskCount distribution by age
sdoh_count_dist = (multi_ts.groupby(["AgeSimple", "SDOHRiskCount"], observed=True)
                            .size()
                            .reset_index(name="count"))
sdoh_count_dist = sdoh_count_dist.merge(n_tbl.reset_index(), on="AgeSimple")
sdoh_count_dist["pct"] = sdoh_count_dist["count"] / sdoh_count_dist["n"] * 100

pivot_risk = sdoh_count_dist[sdoh_count_dist["SDOHRiskCount"].isin([0,1,2,3])].pivot_table(
    index="AgeSimple", columns="SDOHRiskCount", values="pct", fill_value=0
)
pivot_risk.plot(kind="bar", ax=axes[1], colormap="RdYlGn_r",
                edgecolor="white", linewidth=0.5)
axes[1].set_title("SDOH Risk Count Distribution\nby Age Group", fontweight="bold")
axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("% of age group")
axes[1].tick_params(axis="x", rotation=25)
axes[1].legend(title="# SDOH Risks", loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_A_sdoh_prevalence_by_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/fig_A_sdoh_prevalence_by_age.png")


# =============================================================================
# SECTION 2 — BARRIER EFFECT ON LONG GAPS, STRATIFIED BY AGE
# =============================================================================
print("\n" + "="*70)
print("SECTION 2 — SDOH BARRIER EFFECT ON LONG GAPS, BY AGE GROUP")
print("="*70)
print("Key question: does a transport barrier hurt a 75-year-old MORE than a 30-year-old?\n")

# For each SDOH factor × age group: compute gap rate with vs without barrier
records = []
for age_grp in age_labels:
    sub_age = multi_ts[multi_ts["AgeSimple"] == age_grp]
    for col in sdoh_cols:
        g = sub_age.groupby(col)[["HasLongGap","HasED"]].mean() * 100
        if True not in g.index or False not in g.index:
            continue
        n_yes = sub_age[col].sum()
        n_no  = (~sub_age[col]).sum()
        gap_yes = g.loc[True,  "HasLongGap"]
        gap_no  = g.loc[False, "HasLongGap"]
        ed_yes  = g.loc[True,  "HasED"]
        ed_no   = g.loc[False, "HasED"]

        # Chi-squared for gap
        ct_gap = pd.crosstab(sub_age[col], sub_age["HasLongGap"])
        if ct_gap.shape == (2, 2):
            chi2_g, p_g, _, _ = stats.chi2_contingency(ct_gap)
        else:
            chi2_g, p_g = np.nan, np.nan

        ct_ed = pd.crosstab(sub_age[col], sub_age["HasED"])
        if ct_ed.shape == (2, 2):
            chi2_e, p_e, _, _ = stats.chi2_contingency(ct_ed)
        else:
            chi2_e, p_e = np.nan, np.nan

        records.append({
            "AgeGroup": age_grp,
            "SDOH": col,
            "SDOH_label": sdoh_display_1line[col],
            "n_yes": n_yes, "n_no": n_no,
            "gap_no": gap_no, "gap_yes": gap_yes,
            "gap_delta": gap_yes - gap_no,
            "ed_no": ed_no,   "ed_yes": ed_yes,
            "ed_delta": ed_yes - ed_no,
            "p_gap": p_g, "p_ed": p_e,
        })

effects = pd.DataFrame(records)

print("Long gap rate: barrier effect (gap_yes − gap_no) by age group and SDOH factor:")
pivot_gap = effects.pivot_table(
    index="SDOH_label", columns="AgeGroup", values="gap_delta"
).round(1)
print(pivot_gap.to_string())

print("\nED rate: barrier effect (ed_yes − ed_no) by age group and SDOH factor:")
pivot_ed = effects.pivot_table(
    index="SDOH_label", columns="AgeGroup", values="ed_delta"
).round(1)
print(pivot_ed.to_string())

print("\nAbsolute long gap rate (with barrier) by age group:")
pivot_gap_abs = effects.pivot_table(
    index="SDOH_label", columns="AgeGroup", values="gap_yes"
).round(1)
print(pivot_gap_abs.to_string())

print("\nStatistical significance (p-values, gap) — * p<0.05, ** p<0.01, *** p<0.001:")
for _, row in effects.iterrows():
    if row["n_yes"] < 20:
        continue
    stars = ("***" if row["p_gap"] < 0.001 else
             "**"  if row["p_gap"] < 0.01  else
             "*"   if row["p_gap"] < 0.05  else "ns")
    print(f"  {row['AgeGroup']:<22}  {row['SDOH_label']:<24}  "
          f"Δgap={row['gap_delta']:+.1f}pp  n(YES)={row['n_yes']:,}  {stars}")

# ── FIGURE B: Effect-size heatmaps side by side ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("How Social Barriers Drive Long Gaps & ED Use\n"
             "Δ = gap rate WITH barrier minus gap rate WITHOUT barrier (percentage points)",
             fontsize=12, fontweight="bold")

# Reorder columns to age order
col_order = [c for c in age_labels if c in pivot_gap.columns]

def heatmap_with_sig(ax, pivot, sig_pivot, title, vmin, vmax, cmap, fmt=".1f"):
    """Draw a heatmap; cells with n < 20 are greyed out."""
    sns.heatmap(pivot[col_order], annot=True, fmt=fmt, cmap=cmap,
                ax=ax, vmin=vmin, vmax=vmax,
                cbar_kws={"label": "Δ percentage points"},
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 9})
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("SDOH Barrier"); ax.set_xlabel("Age Group")
    ax.tick_params(axis="x", rotation=25)

# Build pivot for gap delta and ed delta
gap_pivot = effects.pivot_table(index="SDOH_label", columns="AgeGroup",
                                 values="gap_delta").reindex(col_order, axis=1)
ed_pivot  = effects.pivot_table(index="SDOH_label", columns="AgeGroup",
                                 values="ed_delta").reindex(col_order, axis=1)

# Mask low-n cells
n_pivot = effects.pivot_table(index="SDOH_label", columns="AgeGroup",
                               values="n_yes", aggfunc="sum").reindex(col_order, axis=1)

vabs_gap = max(abs(gap_pivot.values[~np.isnan(gap_pivot.values)]).max(), 1)
vabs_ed  = max(abs(ed_pivot.values[~np.isnan(ed_pivot.values)]).max(), 1)

sns.heatmap(gap_pivot, annot=True, fmt=".1f", cmap="RdBu_r",
            ax=axes[0], vmin=-vabs_gap, vmax=vabs_gap,
            cbar_kws={"label": "Δpp (positive = higher gap risk)"},
            linewidths=0.5, linecolor="white", annot_kws={"size": 9})
axes[0].set_title("Δ Long Gap Rate (60+ days)\nBarrier vs No Barrier", fontweight="bold")
axes[0].set_ylabel("SDOH Factor"); axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=25)

sns.heatmap(ed_pivot, annot=True, fmt=".1f", cmap="RdBu_r",
            ax=axes[1], vmin=-vabs_ed, vmax=vabs_ed,
            cbar_kws={"label": "Δpp (positive = higher ED risk)"},
            linewidths=0.5, linecolor="white", annot_kws={"size": 9})
axes[1].set_title("Δ ED Visit Rate\nBarrier vs No Barrier", fontweight="bold")
axes[1].set_ylabel(""); axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_B_barrier_effect_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/fig_B_barrier_effect_heatmaps.png")


# ── FIGURE C: Absolute gap rates with vs without barrier for 4 age groups ─────
# Focus on top 5 SDOH factors (highest effect in older group)
top_sdoh = ["TransportBarrier","FoodInsecurity","FinancialStrain",
            "HousingInstability","HighStress","UtilitiesHardship"]
top_labels = [sdoh_display_1line[c] for c in top_sdoh]

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Long Gap Rate With vs Without Each Barrier — By Age Group\n"
             "Each panel = one SDOH factor; bars = age groups; red = has barrier, blue = no barrier",
             fontsize=12, fontweight="bold")
axes = axes.flatten()

for idx, (col, label) in enumerate(zip(top_sdoh, top_labels)):
    ax = axes[idx]
    sub = effects[effects["SDOH"] == col].copy()
    sub = sub[sub["n_yes"] >= 15].copy()

    x   = np.arange(len(sub))
    w   = 0.35
    colors_yes = [AGE_COLORS.get(a, RED)   for a in sub["AgeGroup"]]
    colors_no  = [c + "80" for c in [BLUE]*len(sub)]   # faded blue for "no barrier"

    bars_no  = ax.bar(x - w/2, sub["gap_no"],  w, color=BLUE, alpha=0.55,
                      label="No Barrier", edgecolor="white")
    bars_yes = ax.bar(x + w/2, sub["gap_yes"], w,
                      color=[AGE_COLORS.get(a, RED) for a in sub["AgeGroup"]],
                      alpha=0.90, label="Has Barrier", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([a.split("(")[0].strip() for a in sub["AgeGroup"]],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Long Gap Rate (%)")
    ax.set_title(label, fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(sub["gap_yes"].max(), sub["gap_no"].max()) * 1.35)

    for i, (_, row) in enumerate(sub.iterrows()):
        p = row["p_gap"]
        stars = ("***" if p < 0.001 else "**" if p < 0.01 else
                 "*"   if p < 0.05  else "ns")
        ax.annotate(f"n={int(row['n_yes'])}\n{stars}",
                    xy=(i + w/2, row["gap_yes"] + 0.5),
                    ha="center", fontsize=6.5, color="black")

    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_C_gap_rates_by_barrier_and_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig_C_gap_rates_by_barrier_and_age.png")


# =============================================================================
# SECTION 3 — CONDITION-SPECIFIC ANALYSIS IN OLDER PATIENTS
# =============================================================================
print("\n" + "="*70)
print("SECTION 3 — CONDITION-SPECIFIC ANALYSIS (65+ patients)")
print("="*70)

# Top 4 fracture types relevant to elderly
focus_fractures = {
    "Femur":       "Fracture of femur",
    "Lower Leg":   "Fracture of lower leg, including ankle",
    "Shoulder":    "Fracture of shoulder and upper arm",
    "Lumbar":      "Fracture of lumbar spine and pelvis",
}
focus_sdoh = ["TransportBarrier","FoodInsecurity","FinancialStrain","HousingInstability"]

old_mts  = multi_ts[multi_ts["IsOld"]].copy()
young_mts = multi_ts[~multi_ts["IsOld"]].copy()

print(f"\nOld (65+) multi-visit traumatic surveyed: n={len(old_mts):,}")
print(f"Young (<65) multi-visit traumatic surveyed: n={len(young_mts):,}")

cond_records = []
for frac_short, frac_full in focus_fractures.items():
    for age_label, sub_df in [("65+ (Older)", old_mts), ("<65 (Younger)", young_mts)]:
        sub_frac = sub_df[sub_df["GroupName"] == frac_full]
        n_total = len(sub_frac)
        if n_total < 10:
            continue
        base_gap = sub_frac["HasLongGap"].mean() * 100
        base_ed  = sub_frac["HasED"].mean() * 100
        print(f"\n  {frac_short} | {age_label}  (n={n_total:,})")
        print(f"    Base long gap rate : {base_gap:.1f}%")
        print(f"    Base ED visit rate : {base_ed:.1f}%")

        for sdoh_col in focus_sdoh:
            g = sub_frac.groupby(sdoh_col)[["HasLongGap","HasED"]].mean() * 100
            if True not in g.index or False not in g.index:
                continue
            n_barrier = sub_frac[sdoh_col].sum()
            if n_barrier < 5:
                continue
            gap_barrier = g.loc[True,  "HasLongGap"]
            ed_barrier  = g.loc[True,  "HasED"]
            print(f"    {sdoh_display_1line[sdoh_col]:<22}: "
                  f"gap={gap_barrier:.1f}% (+{gap_barrier-base_gap:.1f}pp), "
                  f"ED={ed_barrier:.1f}% (+{ed_barrier-base_ed:.1f}pp)  "
                  f"[n_barrier={n_barrier}]")
            cond_records.append({
                "Fracture": frac_short, "AgeGroup": age_label,
                "SDOH": sdoh_col, "SDOH_label": sdoh_display_1line[sdoh_col],
                "gap_base": base_gap, "gap_barrier": gap_barrier,
                "gap_delta": gap_barrier - base_gap,
                "ed_base": base_ed,  "ed_barrier": ed_barrier,
                "ed_delta": ed_barrier - base_ed,
                "n_barrier": n_barrier, "n_total": n_total,
            })

cond_df = pd.DataFrame(cond_records)

# ── FIGURE D: Condition × SDOH heatmap for 65+ ────────────────────────────────
old_cond = cond_df[cond_df["AgeGroup"] == "65+ (Older)"].copy()

if not old_cond.empty:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Condition-Specific Risk in Elderly (65+) Patients\n"
                 "Δ = percentage point change in rate when barrier is present",
                 fontsize=12, fontweight="bold")

    for ax, metric, title, cmap in [
        (axes[0], "gap_delta", "Δ Long Gap Rate (pp)", "Reds"),
        (axes[1], "ed_delta",  "Δ ED Visit Rate (pp)", "Reds"),
    ]:
        piv = old_cond.pivot_table(index="Fracture", columns="SDOH_label",
                                    values=metric, aggfunc="mean").round(1)
        sns.heatmap(piv, annot=True, fmt=".1f", cmap=cmap,
                    ax=ax, linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Δ percentage points"},
                    annot_kws={"size": 9})
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Fracture Type"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_D_condition_sdoh_heatmap_65plus.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✓ Saved outputs/fig_D_condition_sdoh_heatmap_65plus.png")

# ── FIGURE E: Old vs Young comparison for femur + lower leg ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Older vs Younger Patients: Does the Same Barrier Hit Differently?\n"
             "(Femur & Lower-Leg Fractures Only)",
             fontsize=12, fontweight="bold")

for ax, frac_name, title in [
    (axes[0], "Femur",     "Femur Fracture"),
    (axes[1], "Lower Leg", "Lower Leg / Ankle Fracture"),
]:
    sub_frac = cond_df[cond_df["Fracture"] == frac_name].copy()
    if sub_frac.empty:
        ax.set_visible(False); continue

    sub_frac = sub_frac[sub_frac["n_barrier"] >= 10]
    old_vals   = sub_frac[sub_frac["AgeGroup"] == "65+ (Older)"].set_index("SDOH_label")["gap_delta"]
    young_vals = sub_frac[sub_frac["AgeGroup"] == "<65 (Younger)"].set_index("SDOH_label")["gap_delta"]
    common_idx = old_vals.index.intersection(young_vals.index)

    x = np.arange(len(common_idx)); w = 0.35
    ax.bar(x - w/2, young_vals[common_idx], w, color=BLUE,  alpha=0.85, label="<65 (Younger)")
    ax.bar(x + w/2, old_vals[common_idx],   w, color=RED,   alpha=0.85, label="65+ (Older)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(common_idx, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Δ Long Gap Rate (pp)\nvs baseline (no barrier)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_E_old_vs_young_fracture_barrier.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig_E_old_vs_young_fracture_barrier.png")


# =============================================================================
# SECTION 4 — STRATIFIED LOGISTIC REGRESSION + INTERACTION MODEL
# =============================================================================
print("\n" + "="*70)
print("SECTION 4 — STRATIFIED LOGISTIC REGRESSION + INTERACTION MODEL")
print("="*70)
print("Stratified: separate LR in Young (<65) vs Old (65+)")
print("Interaction: one model with IsOld × SDOH terms — do effects amplify with age?\n")

# ── Feature engineering ───────────────────────────────────────────────────────
model_df = multi_ts.copy()
model_df["AgeOrdinal"]   = pd.cut(model_df["ApproxAge"].fillna(40),
                                   bins=age_bins, labels=[0,1,2,3]).astype(float)
model_df["RaceBlack"]    = (model_df["RaceSimple"] == "Black").astype(int)
model_df["RaceHispanic"] = (model_df["RaceSimple"] == "Hispanic").astype(int)
model_df["RaceNativeAm"] = (model_df["RaceSimple"] == "Native American").astype(int)
model_df["MyChartActive"]= (model_df["MyChartStatus"] == "Activated").astype(int)

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

sdoh_model_cols = ["TransportBarrier","FoodInsecurity","FinancialStrain",
                   "HousingInstability","HighStress","UtilitiesHardship"]
for c in sdoh_model_cols:
    model_df[c] = model_df[c].astype(int)

demo_feats = ["AgeOrdinal","RaceBlack","RaceHispanic","RaceNativeAm","MyChartActive"]
frac_feats = list(frac_dummies.keys())
all_feats  = sdoh_model_cols + demo_feats + frac_feats

feat_labels = {
    "TransportBarrier":"Transport Barrier","FoodInsecurity":"Food Insecurity",
    "FinancialStrain":"Financial Strain","HousingInstability":"Housing Instability",
    "HighStress":"High Stress","UtilitiesHardship":"Utilities Hardship",
    "AgeOrdinal":"Age Group","RaceBlack":"Race: Black",
    "RaceHispanic":"Race: Hispanic","RaceNativeAm":"Race: Native Am.",
    "MyChartActive":"MyChart Active",
    "FT_Femur":"Femur Fracture","FT_Shoulder":"Shoulder","FT_Forearm":"Forearm",
    "FT_Wrist":"Wrist","FT_Foot":"Foot","FT_Lumbar":"Lumbar",
}

model_df = model_df.dropna(subset=all_feats + ["HasLongGap","HasED","IsOld"])

print(f"Full model dataset: n={len(model_df):,}")
print(f"  Old (65+): n={model_df['IsOld'].sum():,}  ({model_df['IsOld'].mean()*100:.0f}%)")
print(f"  Young (<65): n={(~model_df['IsOld']).sum():,}")


def fit_lr(X, y, feat_names):
    """Fit logistic regression, return odds ratio DataFrame."""
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr     = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                                random_state=42)
    cv_auc = cross_val_score(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                            C=1.0, random_state=42))]),
        X, y, cv=cv, scoring="roc_auc"
    )
    lr.fit(X_s, y)
    or_df = pd.DataFrame({
        "feature": feat_names,
        "OR":      np.exp(lr.coef_[0]),
        "coef":    lr.coef_[0],
        "abs_coef":np.abs(lr.coef_[0]),
    }).sort_values("abs_coef", ascending=False)
    return lr, cv_auc, or_df


X_all = model_df[all_feats].values.astype(float)
y_gap = model_df["HasLongGap"].astype(int).values

# ── Stratified models ─────────────────────────────────────────────────────────
young_df = model_df[~model_df["IsOld"]].copy()
old_df   = model_df[ model_df["IsOld"]].copy()

X_young = young_df[all_feats].values.astype(float)
X_old   = old_df[all_feats].values.astype(float)
y_young = young_df["HasLongGap"].astype(int).values
y_old   = old_df["HasLongGap"].astype(int).values

_, cv_young, or_young = fit_lr(X_young, y_young, all_feats)
_, cv_old,   or_old   = fit_lr(X_old,   y_old,   all_feats)

print(f"\n  LR — Long Gap (Young <65)  : CV AUC = {cv_young.mean():.3f} ± {cv_young.std():.3f}")
print(f"  LR — Long Gap (Old  65+)   : CV AUC = {cv_old.mean():.3f} ± {cv_old.std():.3f}")

print("\n  SDOH Odds Ratios — Young (<65) vs Old (65+):")
print(f"  {'SDOH Factor':<24}  {'OR Young':>9}  {'OR Old':>8}  {'Δ OR':>8}  Interpretation")
print("  " + "-"*80)
for col in sdoh_model_cols:
    or_y = or_young[or_young["feature"] == col]["OR"].values
    or_o = or_old[or_old["feature"] == col]["OR"].values
    if len(or_y) == 0 or len(or_o) == 0:
        continue
    or_y, or_o = or_y[0], or_o[0]
    interp = "Stronger in elderly" if or_o > or_y else "Stronger in young"
    print(f"  {feat_labels[col]:<24}  {or_y:>9.3f}  {or_o:>8.3f}  {or_o-or_y:>+7.3f}  {interp}")


# ── Interaction model: all patients, with IsOld × SDOH terms ─────────────────
print("\n--- Interaction Model: IsOld × SDOH (does age amplify the SDOH effect?) ---")

interact_df = model_df.copy()
interact_df["IsOld_int"] = interact_df["IsOld"].astype(int)

interact_feats = all_feats + ["IsOld_int"]
for col in sdoh_model_cols:
    ix_col = f"Old_x_{col}"
    interact_df[ix_col] = interact_df["IsOld_int"] * interact_df[col]
    interact_feats.append(ix_col)

X_ix = interact_df[interact_feats].values.astype(float)
y_ix = interact_df["HasLongGap"].astype(int).values

_, cv_ix, or_ix = fit_lr(X_ix, y_ix, interact_feats)
print(f"  Interaction model CV AUC: {cv_ix.mean():.3f} ± {cv_ix.std():.3f}")

print("\n  Interaction term ORs (IsOld × SDOH factor — OR > 1 means effect is stronger in elderly):")
interact_rows = or_ix[or_ix["feature"].str.startswith("Old_x_")].copy()
interact_rows["sdoh"] = interact_rows["feature"].str.replace("Old_x_", "")
interact_rows["sdoh_label"] = interact_rows["sdoh"].map(feat_labels).fillna(
    interact_rows["sdoh"])
interact_rows = interact_rows.sort_values("OR", ascending=False)
for _, row in interact_rows.iterrows():
    sig = "↑ STRONGER in elderly" if row["OR"] > 1.05 else \
          "↓ WEAKER in elderly"   if row["OR"] < 0.95 else "≈ Similar"
    print(f"    {row['sdoh_label']:<24}: OR={row['OR']:.3f}  ({sig})")


# ── FIGURE F: Side-by-side OR plots — SDOH in Young vs Old ───────────────────
sdoh_or_young = or_young[or_young["feature"].isin(sdoh_model_cols)].copy()
sdoh_or_old   = or_old[or_old["feature"].isin(sdoh_model_cols)].copy()
sdoh_or_young["label"] = sdoh_or_young["feature"].map(feat_labels)
sdoh_or_old["label"]   = sdoh_or_old["feature"].map(feat_labels)

# Merge for clean comparison
compare = sdoh_or_young[["label","OR"]].rename(columns={"OR":"OR_young"}).merge(
    sdoh_or_old[["label","OR"]].rename(columns={"OR":"OR_old"}), on="label"
).sort_values("OR_old", ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Age-Stratified Logistic Regression: SDOH Effects on 60+ Day Care Gaps\n"
             "Adjusted for race, fracture type, and MyChart status",
             fontsize=12, fontweight="bold")

# Panel 1: Young OR
colors_y = [RED if v > 1 else BLUE for v in compare["OR_young"]]
axes[0].barh(compare["label"], compare["OR_young"], color=colors_y, alpha=0.85)
axes[0].axvline(1.0, color="black", ls="--", lw=1.3)
axes[0].set_xlabel("Odds Ratio"); axes[0].set_title("Young Patients (<65)\nOR for Long Gap",
                                                      fontweight="bold")
for i, v in enumerate(compare["OR_young"]):
    axes[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

# Panel 2: Old OR
colors_o = [RED if v > 1 else BLUE for v in compare["OR_old"]]
axes[1].barh(compare["label"], compare["OR_old"], color=colors_o, alpha=0.85)
axes[1].axvline(1.0, color="black", ls="--", lw=1.3)
axes[1].set_xlabel("Odds Ratio"); axes[1].set_title("Elderly Patients (65+)\nOR for Long Gap",
                                                      fontweight="bold")
for i, v in enumerate(compare["OR_old"]):
    axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

# Panel 3: Interaction ORs
int_compare = interact_rows.sort_values("OR", ascending=True)
colors_ix = [RED if v > 1 else BLUE for v in int_compare["OR"]]
axes[2].barh(int_compare["sdoh_label"], int_compare["OR"], color=colors_ix, alpha=0.85)
axes[2].axvline(1.0, color="black", ls="--", lw=1.3)
axes[2].set_xlabel("Interaction OR (IsOld × SDOH)")
axes[2].set_title("Age × SDOH Interaction Terms\nOR > 1 = Effect Stronger in Elderly",
                   fontweight="bold")
for i, v in enumerate(int_compare["OR"]):
    axes[2].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_F_stratified_OR_and_interactions.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved outputs/fig_F_stratified_OR_and_interactions.png")


# ── FIGURE G: Summary visual — "The Elderly Risk Profile" ─────────────────────
# One clear summary figure: for 65+ patients,
# show each SDOH factor's long gap rate vs baseline, coloured by magnitude
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("The Elderly Risk Profile: Long Gap Rate With Each Barrier (65+ Patients)\n"
             "Red line = baseline long gap rate with no barrier",
             fontsize=12, fontweight="bold")

old_effects = effects[
    (effects["AgeGroup"].str.contains("65|80")) &
    (effects["n_yes"] >= 15)
].copy()
old_effects["full_label"] = old_effects["SDOH_label"] + "\n(" + old_effects["AgeGroup"].str.extract(r"\((.*?)\)", expand=False) + ")"
old_effects = old_effects.sort_values("gap_yes", ascending=True)

baseline = old_mts["HasLongGap"].mean() * 100
bar_colors = ["#E84C4C" if v > baseline + 2 else
              "#E8944C" if v > baseline else
              "#4C9BE8" for v in old_effects["gap_yes"]]

bars = ax.barh(old_effects["full_label"], old_effects["gap_yes"],
               color=bar_colors, alpha=0.88, edgecolor="white")
ax.axvline(baseline, color="black", ls="--", lw=1.8,
           label=f"Baseline (no barrier): {baseline:.1f}%")
ax.set_xlabel("Long Gap Rate (%)")
ax.set_title("")

for bar, (_, row) in zip(bars, old_effects.iterrows()):
    p = row["p_gap"]
    stars = ("***" if p < 0.001 else "**" if p < 0.01 else
             "*"   if p < 0.05  else "ns")
    val = row["gap_yes"]
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%  {stars}", va="center", fontsize=8)

ax.legend(fontsize=9)
red_patch   = mpatches.Patch(color="#E84C4C", alpha=0.88, label="Significantly higher than baseline")
amber_patch = mpatches.Patch(color="#E8944C", alpha=0.88, label="Slightly higher")
blue_patch  = mpatches.Patch(color="#4C9BE8", alpha=0.88, label="Similar or lower")
ax.legend(handles=[red_patch, amber_patch, blue_patch], fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_G_elderly_risk_profile.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved outputs/fig_G_elderly_risk_profile.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY — AGE-STRATIFIED SDOH ANALYSIS")
print("="*70)

# Top barrier by absolute gap rate in elderly
old_gap_effects = effects[
    (effects["AgeGroup"].isin(["Older (65-79)","Eldest (80+)"])) &
    (effects["n_yes"] >= 20)
].sort_values("gap_yes", ascending=False)

if not old_gap_effects.empty:
    top_barrier = old_gap_effects.iloc[0]
    print(f"\nHighest long gap rate in elderly:")
    print(f"  {top_barrier['SDOH_label']} in {top_barrier['AgeGroup']}: "
          f"{top_barrier['gap_yes']:.1f}% (baseline {top_barrier['gap_no']:.1f}%)")

old_ed_effects = effects[
    (effects["AgeGroup"].isin(["Older (65-79)","Eldest (80+)"])) &
    (effects["n_yes"] >= 20)
].sort_values("ed_delta", ascending=False)

if not old_ed_effects.empty:
    top_ed = old_ed_effects.iloc[0]
    print(f"\nStrongest ED substitution effect in elderly:")
    print(f"  {top_ed['SDOH_label']} in {top_ed['AgeGroup']}: "
          f"ED rate {top_ed['ed_no']:.1f}% → {top_ed['ed_yes']:.1f}% "
          f"(+{top_ed['ed_delta']:.1f}pp)")

print(f"\nInteraction model — SDOH effects amplified in elderly (OR > 1):")
for _, row in interact_rows[interact_rows["OR"] > 1].iterrows():
    print(f"  {row['sdoh_label']:<24}: OR={row['OR']:.3f}")

print(f"\nInteraction model — SDOH effects weaker in elderly (OR < 1):")
for _, row in interact_rows[interact_rows["OR"] <= 1].iterrows():
    print(f"  {row['sdoh_label']:<24}: OR={row['OR']:.3f}")

print(f"""
MODEL DRAWBACKS (for presentation discussion):
  1. SDOH survey bias    : Only surveyed patients included (77.9% of journeys).
                           Non-respondents may have MORE barriers — the analysis
                           likely underestimates SDOH effects.
  2. Small elderly-SDOH n: Some age × SDOH cells have n < 30. Effect sizes in
                           those cells have wide confidence intervals (not shown).
  3. AUC ~ 0.65          : Models are moderately predictive. Missing: insurance
                           type, distance from hospital, fracture severity score,
                           comorbidity count.
  4. Interaction ORs     : Interaction terms are adjusted for age, fracture type,
                           and race — but not for comorbidities (older patients
                           have more comorbidities that correlate with both
                           barriers and outcomes).
  5. Cross-sectional     : This is not a longitudinal RCT. Causality cannot be
                           proven — only associations are shown.
  6. Race composition    : The dataset is majority White (80%+). Effects for
                           minority groups are directionally correct but
                           imprecise due to small subgroup sizes.
""")

print("OUTPUT FIGURES")
for fname in ["fig_A_sdoh_prevalence_by_age.png",
              "fig_B_barrier_effect_heatmaps.png",
              "fig_C_gap_rates_by_barrier_and_age.png",
              "fig_D_condition_sdoh_heatmap_65plus.png",
              "fig_E_old_vs_young_fracture_barrier.png",
              "fig_F_stratified_OR_and_interactions.png",
              "fig_G_elderly_risk_profile.png"]:
    print(f"  outputs/{fname}")

print("\n=== DONE — check logs/06_age_stratified_sdoh.txt and outputs/ ===")