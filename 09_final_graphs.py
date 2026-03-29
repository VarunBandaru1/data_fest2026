# =============================================================================
# 09_final_graphs.py  —  DataFest 2026  Final Presentation Figures
# Run after 08_final_analytics.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, sys
from pathlib import Path

warnings.filterwarnings("ignore")

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Check prerequisites ───────────────────────────────────────────────────
required = ["gap_ed_by_age.csv", "sdoh_prevalence_by_age.csv",
            "sdoh_gap_stratified_by_age.csv", "sdoh_ed_stratified_by_age.csv",
            "lr_odds_ratios.csv"]
missing = [f for f in required if not (OUT / f).exists()]
if missing:
    print("ERROR: Run 08_final_analytics.py first. Missing:", missing)
    sys.exit(1)

# ── Shared style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         12,
    "figure.dpi":        160,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
})

BLUE   = "#4C9BE8"
AMBER  = "#E8944C"
RED    = "#E84C4C"
DARK   = "#2C3E50"
LIGHT  = "#F0F4F8"

AGE_ORDER  = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
AGE_LABELS = ["0–17", "18–34", "35–49", "50–64", "65–79", "80+"]
AGE_COLORS = ["#A8D8EA", "#4C9BE8", "#2E6DA4", "#E8944C", "#E84C4C", "#A4200F"]

SDOH_COLS = ["TransportBarrier", "FoodInsecurity", "FinancialStrain",
             "HousingInstability", "HighStress", "UtilitiesHardship"]

# ── Load CSVs ─────────────────────────────────────────────────────────────
gap_ed    = pd.read_csv(OUT / "gap_ed_by_age.csv").set_index("AgeGroup").reindex(AGE_ORDER).reset_index()
prev_age  = pd.read_csv(OUT / "sdoh_prevalence_by_age.csv").set_index("AgeGroup")
gap_strat = pd.read_csv(OUT / "sdoh_gap_stratified_by_age.csv")
ed_strat  = pd.read_csv(OUT / "sdoh_ed_stratified_by_age.csv")
lr        = pd.read_csv(OUT / "lr_odds_ratios.csv")

GAP_PCT = gap_ed["LongGap_pct"].tolist()
n_total = int(gap_ed["LongGap_n"].sum())


# =============================================================================
# SLIDE 1 — Age Gradient
# One message: gap rates triple from young to elderly
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(6)

young_avg   = np.mean([GAP_PCT[1], GAP_PCT[2]])
elderly_avg = np.mean([GAP_PCT[4], GAP_PCT[5]])

bars = ax.bar(x, GAP_PCT, color=AGE_COLORS, width=0.6, edgecolor="white", zorder=3)

# Highlight 50-64 with amber border
ax.bar([3], [GAP_PCT[3]], width=0.6, color="none", edgecolor=AMBER,
       linewidth=3, zorder=4)

# Percentage labels
for bar, val in zip(bars, GAP_PCT):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=DARK)

# Young and elderly baselines
ax.axhline(young_avg,   color=BLUE, linestyle="--", lw=1.8, zorder=2,
           label=f"Young baseline (18–49 avg)  {young_avg:.1f}%")
ax.axhline(elderly_avg, color=RED,  linestyle="--", lw=1.8, zorder=2,
           label=f"Elderly baseline (65+ avg)   {elderly_avg:.1f}%")

# Medicare boundary — simple line + minimal label
ax.axvline(3.5, color=DARK, lw=1.5, zorder=4, linestyle=":")
ax.text(3.55, 2, "← Medicare\n   at 65", fontsize=8.5, color=DARK, va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(AGE_LABELS, fontsize=11)
ax.set_ylabel("Long gap rate (60+ day delay)", fontsize=11)
ax.set_ylim(0, 28)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.set_title("Long Follow-Up Gap Rate by Age Group", fontsize=14, fontweight="bold", pad=12)

fig.text(0.5, -0.02,
         f"Multi-visit traumatic fractures, surveyed patients  |  n = {n_total:,}",
         ha="center", fontsize=9, color="gray")

plt.tight_layout()
fig.savefig(OUT / "slide1_age_gradient.png", bbox_inches="tight")
plt.close()
print("✓ slide1_age_gradient.png")


# =============================================================================
# SLIDE 2 — The Direction Flip
# One message: same barrier raises ED, lowers gaps — for elderly only
# Show as a simple grouped bar chart: Young delta vs Elderly delta
# =============================================================================
young_gap   = gap_strat[gap_strat["AgeGroup"] == "Young (<65)"].set_index("SDOH")
elderly_gap = gap_strat[gap_strat["AgeGroup"] == "Elderly (65+)"].set_index("SDOH")

# Order by young gap delta (largest to smallest)
sdoh_order = young_gap["Delta_pp"].sort_values(ascending=False).index.tolist()
labels = [s.replace(" ", "\n") if len(s) > 12 else s for s in sdoh_order]

y_delta  = young_gap.loc[sdoh_order,   "Delta_pp"].values
e_delta  = elderly_gap.loc[sdoh_order, "Delta_pp"].values

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(sdoh_order))
w = 0.35

b1 = ax.bar(x - w/2, y_delta, w, color=BLUE, label="Young (<65)",   alpha=0.9, zorder=3)
b2 = ax.bar(x + w/2, e_delta, w, color=RED,  label="Elderly (65+)", alpha=0.9, zorder=3)

# Value labels
for bar, val in zip(b1, y_delta):
    sym = "+" if val >= 0 else ""
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + (0.2 if val >= 0 else -0.4),
            f"{sym}{val:.1f}pp", ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.5, color=BLUE, fontweight="bold")

for bar, val in zip(b2, e_delta):
    sym = "+" if val >= 0 else ""
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + (0.2 if val >= 0 else -0.4),
            f"{sym}{val:.1f}pp", ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.5, color=RED, fontweight="bold")

ax.axhline(0, color=DARK, lw=1.2, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Change in long gap rate vs no-barrier baseline (pp)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}pp"))
ax.legend(fontsize=10, framealpha=0.9)
ax.set_title("SDOH Barriers Raise Gap Risk in Young Patients,\nbut Lower It in Elderly Patients",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(axis="y", alpha=0.2)

fig.text(0.5, -0.02,
         "Multi-visit traumatic fractures, surveyed patients",
         ha="center", fontsize=9, color="gray")

plt.tight_layout()
fig.savefig(OUT / "slide2_direction_flip.png", bbox_inches="tight")
plt.close()
print("✓ slide2_direction_flip.png")


# =============================================================================
# SLIDE 3 — Three-Tier Heatmap
# One message: 50-64 carries financial burden; 80+ carries housing isolation
# =============================================================================
TIERS_KEYS   = ["50-64", "65-79", "80+"]
TIERS_LABELS = ["50–64\n(Pre-Medicare)", "65–79\n(Medicare)", "80+\n(Medicare + Medicaid)"]
BARRIER_LABELS = ["Transport", "Food\nInsecurity", "Financial\nStrain",
                  "Housing\nInstability", "High\nStress", "Utilities\nHardship"]

data = prev_age.loc[TIERS_KEYS, SDOH_COLS].values.T   # (6 barriers × 3 tiers)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        color = "white" if val > 6.5 else DARK
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

ax.set_xticks(range(3))
ax.set_xticklabels(TIERS_LABELS, fontsize=11, fontweight="bold")
ax.set_yticks(range(6))
ax.set_yticklabels(BARRIER_LABELS, fontsize=10.5)
ax.tick_params(length=0)
ax.grid(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("% of patients with barrier", fontsize=9)
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax.set_title("SDOH Burden Across Age Tiers", fontsize=14, fontweight="bold", pad=12)

fig.text(0.5, -0.02, "Surveyed patients only", ha="center", fontsize=9, color="gray")

plt.tight_layout()
fig.savefig(OUT / "slide3_heatmap.png", bbox_inches="tight")
plt.close()
print("✓ slide3_heatmap.png")


# =============================================================================
# SLIDE 4 — Forest Plot
# One message: SDOH predicts gaps in young (AUC 0.665), not elderly (AUC 0.535)
# =============================================================================
lr_young   = lr[lr["Model"] == "Young (<65)"].set_index("Feature")
lr_elderly = lr[lr["Model"] == "Elderly (65+)"].set_index("Feature")

FEAT_RENAME = {
    "ApproxAge":          "Age (within group)",
    "UtilitiesHardship":  "Utilities Hardship",
    "FoodInsecurity":     "Food Insecurity",
    "HighStress":         "High Stress",
    "HousingInstability": "Housing Instability",
    "FinancialStrain":    "Financial Strain",
    "TransportBarrier":   "Transport Barrier",
}

feat_order  = lr_young["OR"].sort_values(ascending=True).index.tolist()
feat_labels = [FEAT_RENAME.get(f, f) for f in feat_order]

YOUNG_OR  = lr_young.loc[feat_order,   "OR"].values
YOUNG_LO  = lr_young.loc[feat_order,   "CI_lo"].values
YOUNG_HI  = lr_young.loc[feat_order,   "CI_hi"].values
ELD_OR    = lr_elderly.loc[feat_order, "OR"].values
ELD_LO    = lr_elderly.loc[feat_order, "CI_lo"].values
ELD_HI    = lr_elderly.loc[feat_order, "CI_hi"].values

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(feat_labels))
h = 0.25

ax.errorbar(YOUNG_OR, y + h,
            xerr=[YOUNG_OR - YOUNG_LO, YOUNG_HI - YOUNG_OR],
            fmt="o", color=BLUE, markersize=8, capsize=4, lw=2,
            label=f"Young (<65)    AUC = 0.665")

ax.errorbar(ELD_OR, y - h,
            xerr=[ELD_OR - ELD_LO, ELD_HI - ELD_OR],
            fmt="s", color=RED, markersize=8, capsize=4, lw=2,
            label=f"Elderly (65+)  AUC = 0.535")

ax.axvline(1.0, color=DARK, lw=1.2, linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(feat_labels, fontsize=10.5)
ax.set_xlabel("Odds Ratio  (log scale)", fontsize=11)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="x", alpha=0.2)
ax.set_title("SDOH Factors Predict Gaps in Young Patients — Not in Elderly",
             fontsize=13, fontweight="bold", pad=12)

fig.text(0.5, -0.02,
         "Logistic regression, multi-visit journeys, surveyed patients  |  OR > 1 = higher gap risk",
         ha="center", fontsize=9, color="gray")

plt.tight_layout()
fig.savefig(OUT / "slide4_forest_plot.png", bbox_inches="tight")
plt.close()
print("✓ slide4_forest_plot.png")

print("\nAll done — outputs/slide1 through slide4 saved.")
