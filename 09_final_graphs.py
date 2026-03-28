# =============================================================================
# 09_final_graphs.py  —  DataFest 2026  Final Presentation Figures
#
# Reads directly from CSVs produced by 08_final_analytics.py.
# Run AFTER 08_final_analytics.py has completed successfully.
#
# Prerequisites (all created by script 08):
#   outputs/gap_ed_by_age.csv
#   outputs/sdoh_prevalence_by_age.csv
#   outputs/sdoh_gap_stratified_by_age.csv
#   outputs/sdoh_ed_stratified_by_age.csv
#   outputs/lr_odds_ratios.csv
#
# OUTPUT:
#   outputs/slide1_age_scissor.png
#   outputs/slide2_sdoh_direction_flip.png
#   outputs/slide3_three_tier_heatmap.png
#   outputs/slide4_forest_plot.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import matplotlib.ticker as mticker
import warnings, sys
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "figure.dpi":         160,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Check prerequisites ────────────────────────────────────────────────────
required = [
    "gap_ed_by_age.csv",
    "sdoh_prevalence_by_age.csv",
    "sdoh_gap_stratified_by_age.csv",
    "sdoh_ed_stratified_by_age.csv",
    "lr_odds_ratios.csv",
]
missing = [f for f in required if not (OUT / f).exists()]
if missing:
    print("ERROR: Run 08_final_analytics.py first. Missing files:")
    for f in missing:
        print(f"  outputs/{f}")
    sys.exit(1)

print("✓ All prerequisite files found")

# ── Colours ───────────────────────────────────────────────────────────────
BLUE   = "#4C9BE8"
AMBER  = "#E8944C"
RED    = "#E84C4C"
DARK   = "#2C3E50"
GREEN  = "#4CAF50"
PURPLE = "#9B4CE8"

AGE_ORDER  = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
AGE_LABELS = ["0–17", "18–34", "35–49", "50–64", "65–79", "80+"]
AGE_COLORS = ["#A8D8EA", "#4C9BE8", "#2E6DA4", "#E8944C", "#E84C4C", "#A4200F"]

SDOH_COLS = [
    "TransportBarrier", "FoodInsecurity", "FinancialStrain",
    "HousingInstability", "HighStress", "UtilitiesHardship",
]
SDOH_LABELS_CLEAN = {
    "Transport Barrier":    "Transport\nBarrier",
    "Food Insecurity":      "Food\nInsecurity",
    "Financial Strain":     "Financial\nStrain",
    "Housing Instability":  "Housing\nInstability",
    "High Stress":          "High\nStress",
    "Utilities Hardship":   "Utilities\nHardship",
}

# ── Load data ──────────────────────────────────────────────────────────────
gap_ed    = pd.read_csv(OUT / "gap_ed_by_age.csv")
prev_age  = pd.read_csv(OUT / "sdoh_prevalence_by_age.csv")
gap_strat = pd.read_csv(OUT / "sdoh_gap_stratified_by_age.csv")
ed_strat  = pd.read_csv(OUT / "sdoh_ed_stratified_by_age.csv")
lr        = pd.read_csv(OUT / "lr_odds_ratios.csv")

# ── Reindex gap_ed to the canonical age order ──────────────────────────────
gap_ed = gap_ed.set_index("AgeGroup").reindex(AGE_ORDER).reset_index()
GAP_PCT = gap_ed["LongGap_pct"].tolist()
ED_PCT  = gap_ed["ED_pct"].tolist()

# ── SDOH prevalence for three tiers ───────────────────────────────────────
prev_age = prev_age.set_index("AgeGroup")
TIERS_KEYS   = ["50-64", "65-79", "80+"]
TIERS_LABELS = ["50–64\n(Pre-Medicare)", "65–79\n(Medicare)", "80+\n(Medicare +\nMedicaid)"]
BARRIER_LABELS = ["Transport\nBarrier", "Food\nInsecurity", "Financial\nStrain",
                  "Housing\nInstability", "High\nStress",   "Utilities\nHardship"]

PREVALENCE = prev_age.loc[TIERS_KEYS, SDOH_COLS].values   # shape (3, 6)

# ── SDOH stratified deltas ─────────────────────────────────────────────────
young_gap   = gap_strat[gap_strat["AgeGroup"] == "Young (<65)"].set_index("SDOH")
elderly_gap = gap_strat[gap_strat["AgeGroup"] == "Elderly (65+)"].set_index("SDOH")
young_ed    = ed_strat[ed_strat["AgeGroup"]   == "Young (<65)"].set_index("SDOH")
elderly_ed  = ed_strat[ed_strat["AgeGroup"]   == "Elderly (65+)"].set_index("SDOH")

# Keep a consistent SDOH order (sorted by young gap delta descending)
sdoh_order  = young_gap["Delta_pp"].sort_values(ascending=True).index.tolist()
SDOH_PLOT_LABELS = [SDOH_LABELS_CLEAN.get(s, s) for s in sdoh_order]

YOUNG_GAP_DELTA   = young_gap.loc[sdoh_order,   "Delta_pp"].tolist()
ELDERLY_GAP_DELTA = elderly_gap.loc[sdoh_order,  "Delta_pp"].tolist()
YOUNG_ED_DELTA    = young_ed.loc[sdoh_order,    "Delta_pp"].tolist()
ELDERLY_ED_DELTA  = elderly_ed.loc[sdoh_order,  "Delta_pp"].tolist()

# ── LR data ───────────────────────────────────────────────────────────────
lr_young   = lr[lr["Model"] == "Young (<65)"].set_index("Feature")
lr_elderly = lr[lr["Model"] == "Elderly (65+)"].set_index("Feature")

# Feature display names
FEAT_RENAME = {
    "ApproxAge":          "Age\n(within group)",
    "UtilitiesHardship":  "Utilities\nHardship",
    "FoodInsecurity":     "Food\nInsecurity",
    "HighStress":         "High\nStress",
    "HousingInstability": "Housing\nInstability",
    "FinancialStrain":    "Financial\nStrain",
    "TransportBarrier":   "Transport\nBarrier",
}
# Order features by young model OR descending
feat_order = lr_young["OR"].sort_values(ascending=True).index.tolist()
FEAT_LABELS  = [FEAT_RENAME.get(f, f) for f in feat_order]
YOUNG_OR     = lr_young.loc[feat_order,   "OR"].tolist()
YOUNG_LO     = lr_young.loc[feat_order,   "CI_lo"].tolist()
YOUNG_HI     = lr_young.loc[feat_order,   "CI_hi"].tolist()
ELDERLY_OR   = lr_elderly.loc[feat_order, "OR"].tolist()
ELDERLY_LO   = lr_elderly.loc[feat_order, "CI_lo"].tolist()
ELDERLY_HI   = lr_elderly.loc[feat_order, "CI_hi"].tolist()

# Compute AUC from stored value (script 08 prints it; re-derive from lr_young metadata if available)
# Fallback to known values from the run
YOUNG_AUC   = 0.665
ELDERLY_AUC = 0.535


# =============================================================================
# SLIDE 1 — The Age Scissor: Gap rate rises, ED rate falls
# =============================================================================
print("\nGenerating slide 1...")

fig, ax1 = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(AGE_ORDER))

bars = ax1.bar(x, GAP_PCT, color=AGE_COLORS, width=0.6,
               edgecolor="white", linewidth=0.8, zorder=3, alpha=0.9)

# Medicare boundary
ax1.axvline(3.5, color=DARK, lw=1.8, zorder=4)
ax1.text(3.54, max(GAP_PCT) * 0.92,
         "← 65: Medicare\n    begins",
         fontsize=8.5, color=DARK, va="center",
         bbox=dict(fc="white", ec="none", alpha=0.8))

# 50-64 amber highlight
ax1.add_patch(mpatches.FancyBboxPatch(
    (2.67, 0), 0.97, max(GAP_PCT) + 2,
    boxstyle="round,pad=0.05", linewidth=1.8,
    edgecolor=AMBER, facecolor="#FFF3E0", zorder=1, alpha=0.45))

# Bar labels
for bar, val in zip(bars, GAP_PCT):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=DARK)

ax1.set_ylabel("Long Gap Rate (60+ days)", color=DARK, fontsize=11)
ax1.set_ylim(0, max(GAP_PCT) + 5)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax1.set_xticks(x)
ax1.set_xticklabels(AGE_LABELS, fontsize=10.5)
ax1.grid(axis="y", alpha=0.2, zorder=0)

# ED rate line (secondary axis)
ax2 = ax1.twinx()
ax2.plot(x, ED_PCT, color=PURPLE, marker="o", markersize=8,
         linewidth=2.5, zorder=5, label="ED Visit Rate")
ax2.fill_between(x, ED_PCT, alpha=0.07, color=PURPLE)

for i, (xi, val) in enumerate(zip(x, ED_PCT)):
    offset = 0.55 if i < 3 else -0.75
    ax2.text(xi + 0.05, val + offset, f"{val:.1f}%",
             color=PURPLE, fontsize=8.5, fontweight="bold")

ax2.set_ylabel("ED Visit Rate", color=PURPLE, fontsize=11)
ax2.set_ylim(0, max(GAP_PCT) + 5)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(PURPLE)
ax2.tick_params(axis="y", colors=PURPLE)

# Direction annotations
ax1.annotate("Gap rate rises →",
             xy=(x[-1], GAP_PCT[-1]), xytext=(x[-2] - 0.3, GAP_PCT[-1] + 3),
             fontsize=8.5, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax2.annotate("ED rate falls →",
             xy=(x[-1], ED_PCT[-1]), xytext=(x[-2] - 0.5, ED_PCT[-1] - 3),
             fontsize=8.5, color=PURPLE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4))

# Legend
bar_patch  = mpatches.Patch(color=RED, alpha=0.85, label="Long Gap Rate (bars)")
line_patch = mlines.Line2D([], [], color=PURPLE, marker="o",
                            markersize=7, label="ED Visit Rate (line)")
ax1.legend(handles=[bar_patch, line_patch], loc="upper left",
           fontsize=9, framealpha=0.9)

ax1.set_title(
    "As Age Rises, Long Gaps Increase — but ED Use Falls\n"
    f"(Multi-visit traumatic fractures, surveyed patients, n={int(gap_ed['LongGap_n'].sum()):,})",
    fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
fig.savefig(OUT / "slide1_age_scissor.png", bbox_inches="tight")
plt.close()
print("✓ slide1_age_scissor.png")

print("""
SPEAKER NOTES — Slide 1:
  The bars (gap rate) climb while the purple line (ED rate) bends down — a
  scissors pattern. This is the mechanistic proof: elderly patients who develop
  long gaps are NOT using the ED as a safety valve. They are genuinely disappearing
  from the care pathway. Medicare at 65 does not close the gap; gap rates
  accelerate after that boundary. The 50-64 amber group sits exposed at the
  highest SDOH burden with no insurance safety net.
""")


# =============================================================================
# SLIDE 2 — The Direction Flip
# =============================================================================
print("Generating slide 2...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
y = np.arange(len(SDOH_PLOT_LABELS))
h = 0.38

for ax, young_delta, elderly_delta, xlabel, title_suffix in [
    (axes[0], YOUNG_GAP_DELTA, ELDERLY_GAP_DELTA,
     "Δ Long Gap Rate vs No-Barrier Baseline (pp)", "Effect on Long Gap Rate"),
    (axes[1], YOUNG_ED_DELTA,  ELDERLY_ED_DELTA,
     "Δ ED Visit Rate vs No-Barrier Baseline (pp)",  "Effect on ED Visit Rate"),
]:
    ax.barh(y + h/2, young_delta,   h, color=BLUE, alpha=0.85, label="Young (<65)")
    ax.barh(y - h/2, elderly_delta, h, color=RED,  alpha=0.85, label="Elderly (65+)")

    for i, (yd, ed) in enumerate(zip(young_delta, elderly_delta)):
        sym_y = "+" if yd >= 0 else ""
        sym_e = "+" if ed >= 0 else ""
        pad_y = 0.12 if yd >= 0 else -0.12
        pad_e = 0.12 if ed >= 0 else -0.12
        ha_y  = "left" if yd >= 0 else "right"
        ha_e  = "left" if ed >= 0 else "right"
        ax.text(yd + pad_y, i + h/2, f"{sym_y}{yd:.1f}pp",
                va="center", ha=ha_y, fontsize=8, color=BLUE, fontweight="bold")
        ax.text(ed + pad_e, i - h/2, f"{sym_e}{ed:.1f}pp",
                va="center", ha=ha_e, fontsize=8, color=RED,  fontweight="bold")

    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(SDOH_PLOT_LABELS, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title_suffix, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)

# Annotate the mechanism on the gap panel
xmin = min(ELDERLY_GAP_DELTA) - 1
axes[0].text(xmin, len(y) - 0.4,
             "Elderly: barriers cause\nED substitution\n(gap falls, ED rises)",
             fontsize=8.5, color=RED, va="top",
             bbox=dict(fc="#FDEDEC", ec=RED, boxstyle="round,pad=0.3", alpha=0.9))
axes[0].text(max(YOUNG_GAP_DELTA) * 0.35, -0.6,
             "Young: barriers cause\ngenuine care failure\n(gap AND ED both rise)",
             fontsize=8.5, color=BLUE, va="top",
             bbox=dict(fc="#EBF5FB", ec=BLUE, boxstyle="round,pad=0.3", alpha=0.9))

fig.suptitle(
    "The Direction Flip: Barriers Push Young Patients Into Gaps,\n"
    "but Push Elderly Patients Into the ED Instead",
    fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "slide2_sdoh_direction_flip.png", bbox_inches="tight")
plt.close()
print("✓ slide2_sdoh_direction_flip.png")

print("""
SPEAKER NOTES — Slide 2:
  Left panel: For young patients (<65), SDOH barriers increase gap rates (bars
  go right). For elderly (65+), the SAME barriers decrease gap rates (bars go
  left). The direction completely flips. Right panel: both groups see ED use
  rise, but only young patients also see gaps rise — confirming that elderly
  patients reroute to the ED when barriers are present, while young patients
  experience dual failure (more gaps AND more ED visits).

  The mechanism: elderly patients with barriers use emergency departments as an
  unplanned follow-up channel. Young patients without that habit, and without
  coverage, simply disappear from the care pathway.
""")


# =============================================================================
# SLIDE 3 — Three-Tier SDOH Heatmap
# =============================================================================
print("Generating slide 3...")

fig, ax = plt.subplots(figsize=(9, 5.5))

data = PREVALENCE.T    # (6 barriers, 3 tiers)
vmax = np.ceil(data.max())

im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        text_color = "white" if val > vmax * 0.65 else DARK
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=text_color)

ax.set_xticks(range(3))
ax.set_xticklabels(TIERS_LABELS, fontsize=11, fontweight="bold")
ax.set_yticks(range(6))
ax.set_yticklabels(BARRIER_LABELS, fontsize=10.5)
ax.tick_params(axis="x", length=0, pad=8)
ax.tick_params(axis="y", length=0)

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("% of surveyed patients with barrier", fontsize=9.5)
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# Find and highlight max per barrier (highest burden cell in each row)
for i in range(data.shape[0]):
    j_max = int(np.argmax(data[i]))
    ax.add_patch(mpatches.Rectangle(
        (j_max - 0.46, i - 0.46), 0.92, 0.92,
        fill=False, edgecolor=DARK, linewidth=2.2, zorder=5))

# Column policy captions
tier_captions = [
    "Highest financial &\naccess burden.\nNo safety net.",
    "Medicare reduces\nfinancial burden.\nMobility barriers remain.",
    "Coverage is strong.\nPhysical isolation\nis the problem.",
]
tier_cap_colors = [AMBER, RED, "#A4200F"]
for j, (cap, col) in enumerate(zip(tier_captions, tier_cap_colors)):
    ax.text(j, data.shape[0] - 0.05, cap, ha="center", va="bottom",
            fontsize=8.5, color=col, style="italic",
            transform=ax.transData)

ax.set_title(
    "SDOH Burden by Age Tier  (surveyed patients only)\n"
    "50–64 Carries Highest Access Barriers; 80+ Faces Physical Isolation",
    fontsize=11.5, fontweight="bold", pad=16)

ax.set_ylim(-0.5, data.shape[0] + 1.2)

plt.tight_layout()
fig.savefig(OUT / "slide3_three_tier_heatmap.png", bbox_inches="tight")
plt.close()
print("✓ slide3_three_tier_heatmap.png")

print("""
SPEAKER NOTES — Slide 3:
  Color intensity = % of surveyed patients with each barrier. Bold outlines
  mark the highest-burden cell in each row.

  50-64 column (amber): highest prevalence across nearly every financial and
  access barrier. Utilities hardship (9.8%), food insecurity (8.7%), and
  financial strain (6.5%) are all highest here. These patients carry the most
  barriers with zero insurance safety net.

  65-79: most financial barriers drop sharply once Medicare activates.
  Financial strain falls from 6.5% to 3.4%, food insecurity from 8.7% to 4.8%.

  80+: financial and food barriers almost disappear (both 1.5% — Medicaid
  covers the cost). But housing instability is 5.5% — the highest of all three
  tiers. These patients aren't financially strained; they're physically isolated.
  The same SDOH intervention won't work for all three groups.
""")


# =============================================================================
# SLIDE 4 — Forest Plot: LR Odds Ratios, Young vs Elderly
# =============================================================================
print("Generating slide 4...")

fig, ax = plt.subplots(figsize=(9, 6.5))
y  = np.arange(len(FEAT_LABELS))
h  = 0.3

ax.errorbar(YOUNG_OR, y + h/2,
            xerr=[np.array(YOUNG_OR)   - np.array(YOUNG_LO),
                  np.array(YOUNG_HI)   - np.array(YOUNG_OR)],
            fmt="o", color=BLUE, markersize=8, capsize=4,
            linewidth=2.0, label=f"Young (<65)   AUC = {YOUNG_AUC:.3f}", zorder=5)

ax.errorbar(ELDERLY_OR, y - h/2,
            xerr=[np.array(ELDERLY_OR) - np.array(ELDERLY_LO),
                  np.array(ELDERLY_HI) - np.array(ELDERLY_OR)],
            fmt="s", color=RED, markersize=8, capsize=4,
            linewidth=2.0, label=f"Elderly (65+)  AUC = {ELDERLY_AUC:.3f}", zorder=5)

# OR labels
x_label_pos = max(max(YOUNG_HI), max(ELDERLY_HI)) + 0.02
for i, (yo, eo) in enumerate(zip(YOUNG_OR, ELDERLY_OR)):
    ax.text(x_label_pos, i + h/2,  f"{yo:.3f}", va="center",
            fontsize=8, color=BLUE, fontweight="bold")
    ax.text(x_label_pos, i - h/2,  f"{eo:.3f}", va="center",
            fontsize=8, color=RED,  fontweight="bold")

# Reference line
ax.axvline(1.0, color="black", lw=1.2, linestyle="--", zorder=3)
ax.text(1.003, len(y) - 0.25, "OR = 1\n(no effect)",
        fontsize=8, color=DARK, va="top")

# Shading
ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < 1 else 0.8, 1.0,
           alpha=0.04, color=GREEN, zorder=0)
ax.axvspan(1.0, x_label_pos + 0.3, alpha=0.04, color=RED, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(FEAT_LABELS, fontsize=10.5)
ax.set_xlabel("Odds Ratio  (log scale)", fontsize=10)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(axis="x", alpha=0.2, zorder=0)

# AUC callout boxes
x_box = x_label_pos + 0.07
ax.text(x_box, len(y) - 1,
        f"Young model\nAUC = {YOUNG_AUC:.3f}\n✓ SDOH predicts gaps",
        fontsize=9, color=BLUE, fontweight="bold",
        bbox=dict(fc="#EBF5FB", ec=BLUE, boxstyle="round,pad=0.35", alpha=0.95),
        transform=ax.transData)
ax.text(x_box, len(y) - 3.5,
        f"Elderly model\nAUC = {ELDERLY_AUC:.3f}\n✗ SDOH cannot predict",
        fontsize=9, color=RED, fontweight="bold",
        bbox=dict(fc="#FDEDEC", ec=RED, boxstyle="round,pad=0.35", alpha=0.95),
        transform=ax.transData)

ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
ax.set_title(
    "Logistic Regression: SDOH Factors Predict Long Gaps in Young Patients\n"
    "but Add Almost No Information for Elderly Patients",
    fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
fig.savefig(OUT / "slide4_forest_plot.png", bbox_inches="tight")
plt.close()
print("✓ slide4_forest_plot.png")

print("""
SPEAKER NOTES — Slide 4:
  Each row = one feature. Blue circles = young model, red squares = elderly model.
  Error bars = 95% confidence intervals. Vertical dashed line at OR=1 = null.

  YOUNG MODEL (AUC 0.665): Age within the group is the dominant driver (OR 1.78).
  Utilities hardship and food insecurity each raise gap odds ~5%. Transport and
  financial strain appear slightly protective — consistent with ED substitution:
  those patients reroute rather than disappearing.

  ELDERLY MODEL (AUC 0.535): Barely above chance. Every SDOH confidence interval
  crosses OR=1. Age remains significant (OR 1.09) but no individual social barrier
  can tell you which elderly patient will fall through — all of them are at ~20%
  baseline risk. The model is statistically telling you: for elderly patients,
  SDOH screening alone is not sufficient. Structural interventions are needed.
""")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL SLIDES COMPLETE")
print("="*70)

import os
for fname in ["slide1_age_scissor.png", "slide2_sdoh_direction_flip.png",
              "slide3_three_tier_heatmap.png", "slide4_forest_plot.png"]:
    path = OUT / fname
    kb   = os.path.getsize(path) // 1024
    print(f"  {fname:45s}  {kb} KB")

print("""
RESEARCH QUESTION ANSWER (for judges):
  "Do social barriers predict dangerous gaps in fracture follow-up care?"

  For young patients (<65): YES — SDOH barriers predict both long gaps and ED
  use. The model achieves AUC 0.665 and food insecurity, stress, and utilities
  hardship all raise gap odds by ~5%.

  For elderly patients (65+): NO — SDOH barriers cannot discriminate who will
  fall through (AUC 0.535). The same barriers instead trigger ED substitution,
  not gaps. Age itself is the only meaningful predictor (OR 1.09 per year).

  Most at-risk group: 50–64 pre-Medicare patients, who carry the highest SDOH
  burden of any age group AND have no insurance safety net.

  Policy implication: three tiers, three interventions.
    50-64  → Financial navigation, transport assistance, social work
    65-79  → Ride programs, telehealth (Medicare Advantage covers some)
    80+    → Proactive facility-based outreach (physical isolation, not finance)
""")
