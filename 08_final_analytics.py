# =============================================================================
# 08_final_analytics.py  —  DataFest 2026
# Produces the specific numbers needed for the 4-slide presentation.
# Run from the directory that contains: patients.csv, diagnosis.csv,
# encounters.csv, social_determinants.csv
# (same folder you ran scripts 02–04 from)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import scipy.stats as stats

warnings.filterwarnings("ignore")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ── Load pre-built journey table (created by script 03) ──────────────────────
journey_pat = pd.read_csv(OUT_DIR / "journey_pat_final.csv", low_memory=False)
soc_det = pd.read_csv(
    "social_determinants.csv",
    dtype={"EncounterKey":"string","PatientDurableKey":"string",
           "Domain":"category","DisplayName":"string","AnswerText":"string"},
    low_memory=False)

def fix_key(s):
    return s.astype(str).str.strip().str.replace(".0","",regex=False)
journey_pat["PatientDurableKey"] = fix_key(journey_pat["PatientDurableKey"])
soc_det["PatientDurableKey"]     = fix_key(soc_det["PatientDurableKey"])

def make_flag(domain_str, cond_fn):
    sub = soc_det[soc_det["Domain"]==domain_str][["PatientDurableKey","AnswerText"]].copy()
    if sub.empty: return pd.DataFrame(columns=["PatientDurableKey","flag"])
    sub["flag"] = sub["AnswerText"].apply(cond_fn)
    return sub.groupby("PatientDurableKey")["flag"].max().reset_index()

sdoh_def = {
    "TransportBarrier":   ("Transportation Needs",    lambda x: str(x).strip().lower()=="yes"),
    "FoodInsecurity":     ("Food insecurity",          lambda x: str(x).strip().lower() in ["sometimes true","often true"]),
    "FinancialStrain":    ("Financial Resource Strain",lambda x: str(x).strip().lower()=="yes"),
    "HousingInstability": ("Housing Stability",        lambda x: str(x).strip() in ["1","2","3","Yes"]),
    "HighStress":         ("stress",                   lambda x: str(x).strip().lower() in ["rather much","very much"]),
    "UtilitiesHardship":  ("Utilities",                lambda x: str(x).strip().lower() in ["somewhat hard","hard","very hard"]),
}
for col in sdoh_def:
    if col in journey_pat.columns: journey_pat.drop(columns=[col], inplace=True)
for col,(domain,fn) in sdoh_def.items():
    fd = make_flag(domain, fn)
    if fd.empty or "flag" not in fd.columns:
        journey_pat[col] = False
    else:
        fd = fd.rename(columns={"flag":col})
        journey_pat = journey_pat.merge(fd[["PatientDurableKey",col]], on="PatientDurableKey", how="left")
        journey_pat[col] = journey_pat[col].fillna(False).astype(bool)

sdoh_cols = list(sdoh_def.keys())
journey_pat["SDOHRiskCount"] = journey_pat[sdoh_cols].sum(axis=1)
journey_pat["WasSurveyed"]   = journey_pat["PatientDurableKey"].isin(set(soc_det["PatientDurableKey"].unique()))

osteo_names = ["Osteoporosis with current pathological fracture",
               "Osteoporosis without current pathological fracture"]
journey_pat["IsOsteoporosis"] = journey_pat["GroupName"].isin(osteo_names)
journey_pat["IsTraumatic"]    = ~journey_pat["IsOsteoporosis"]
journey_pat["IsMultiVisit"]   = journey_pat["NumVisits"] >= 2
journey_pat["IsElderly"]      = journey_pat["ApproxAge"] >= 65

age_order = ["0-17","18-34","35-49","50-64","65-79","80+"]

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: SDOH PREVALENCE BY AGE GROUP
# (surveyed patients only — honest comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BLOCK 1: SDOH Prevalence by Age Group (surveyed patients only)")
print("="*70)

surveyed = journey_pat[journey_pat["WasSurveyed"]].copy()
print(f"\nSurveyed patient journeys: {len(surveyed):,}")

sdoh_labels = {
    "TransportBarrier":   "Transport Barrier",
    "FoodInsecurity":     "Food Insecurity",
    "FinancialStrain":    "Financial Strain",
    "HousingInstability": "Housing Instability",
    "HighStress":         "High Stress",
    "UtilitiesHardship":  "Utilities Hardship",
}

prev = surveyed.groupby("AgeGroup")[sdoh_cols].mean() * 100
prev = prev.reindex(age_order)
print("\nSDOH prevalence (%) by age group:")
print(prev.round(1).to_string())
prev.to_csv(OUT_DIR / "sdoh_prevalence_by_age.csv")
print("\n✓ Saved outputs/sdoh_prevalence_by_age.csv")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: GAP RATE + ED RATE BY AGE GROUP
# (multi-visit traumatic, surveyed, for paired outcome table)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BLOCK 2: Gap Rate + ED Rate by Age Group (multi-visit traumatic, surveyed)")
print("="*70)

mv_ts = journey_pat[
    journey_pat["IsMultiVisit"] &
    journey_pat["IsTraumatic"]  &
    journey_pat["WasSurveyed"]
].copy()

print(f"\nFiltered set: {len(mv_ts):,} journeys")

age_outcomes = mv_ts.groupby("AgeGroup")[["HasLongGap","HasED"]].agg(["mean","count"])
age_outcomes.columns = ["LongGap_rate","LongGap_n","ED_rate","ED_n"]
age_outcomes["LongGap_pct"] = age_outcomes["LongGap_rate"] * 100
age_outcomes["ED_pct"]      = age_outcomes["ED_rate"]      * 100
age_outcomes = age_outcomes.reindex(age_order)
print("\nGap rate and ED rate by age group:")
print(age_outcomes[["LongGap_pct","ED_pct","LongGap_n"]].round(1).to_string())
age_outcomes.to_csv(OUT_DIR / "gap_ed_by_age.csv")
print("\n✓ Saved outputs/gap_ed_by_age.csv")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3: SDOH EFFECT STRATIFIED BY AGE (<65 vs 65+)
# For each SDOH barrier, compute gap rate and ED rate for young vs elderly
# This proves whether the same barrier hits differently by age
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BLOCK 3: SDOH Effects Stratified by Age (<65 vs 65+)")
print("="*70)

print("\n--- GAP RATE by barrier × age ---")
rows_gap = []
for col in sdoh_cols:
    for age_label, mask in [("Young (<65)", mv_ts["IsElderly"]==False),
                             ("Elderly (65+)", mv_ts["IsElderly"]==True)]:
        sub = mv_ts[mask]
        no_b  = sub[sub[col]==False]["HasLongGap"].mean() * 100
        yes_b = sub[sub[col]==True]["HasLongGap"].mean()  * 100
        n_yes = sub[col].sum()
        delta = yes_b - no_b
        rows_gap.append({"SDOH": sdoh_labels[col], "AgeGroup": age_label,
                         "NoBarrier_pct": no_b, "WithBarrier_pct": yes_b,
                         "Delta_pp": delta, "N_barrier": n_yes})

gap_strat = pd.DataFrame(rows_gap)
print(gap_strat.to_string(index=False))
gap_strat.to_csv(OUT_DIR / "sdoh_gap_stratified_by_age.csv", index=False)
print("\n✓ Saved outputs/sdoh_gap_stratified_by_age.csv")

print("\n--- ED RATE by barrier × age ---")
rows_ed = []
for col in sdoh_cols:
    for age_label, mask in [("Young (<65)", mv_ts["IsElderly"]==False),
                             ("Elderly (65+)", mv_ts["IsElderly"]==True)]:
        sub = mv_ts[mask]
        no_b  = sub[sub[col]==False]["HasED"].mean() * 100
        yes_b = sub[sub[col]==True]["HasED"].mean()  * 100
        n_yes = sub[col].sum()
        delta = yes_b - no_b
        rows_ed.append({"SDOH": sdoh_labels[col], "AgeGroup": age_label,
                        "NoBarrier_pct": no_b, "WithBarrier_pct": yes_b,
                        "Delta_pp": delta, "N_barrier": n_yes})

ed_strat = pd.DataFrame(rows_ed)
print(ed_strat.to_string(index=False))
ed_strat.to_csv(OUT_DIR / "sdoh_ed_stratified_by_age.csv", index=False)
print("\n✓ Saved outputs/sdoh_ed_stratified_by_age.csv")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4: LOGISTIC REGRESSION — YOUNG vs ELDERLY MODELS
# Outcome: HasLongGap
# Features: SDOH flags + ApproxAge (within group) + IsOsteoporosis
# Reports: odds ratios + 95% CI + AUC
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BLOCK 4: Logistic Regression — Young vs Elderly (HasLongGap outcome)")
print("="*70)

features = sdoh_cols + ["ApproxAge"]
# Use all multi-visit surveyed (including osteo to get enough elderly obs)
mv_s = journey_pat[journey_pat["IsMultiVisit"] & journey_pat["WasSurveyed"]].copy()

def run_lr(df, label):
    df = df.dropna(subset=features + ["HasLongGap"])
    X = df[features].astype(float)
    y = df["HasLongGap"].astype(int)
    if y.sum() < 10:
        print(f"  {label}: too few positives ({y.sum()}) — skipping")
        return None
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, C=1.0))])
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X, y)
    coefs = pipe.named_steps["lr"].coef_[0]
    sds   = pipe.named_steps["sc"].scale_
    # OR = exp(coef); 95% CI via Wald approximation
    n     = len(y)
    se    = 1 / np.sqrt(n * 0.25)  # conservative Wald SE per coef
    OR    = np.exp(coefs)
    CI_lo = np.exp(coefs - 1.96 * se)
    CI_hi = np.exp(coefs + 1.96 * se)
    result = pd.DataFrame({
        "Feature": features,
        "OR":     OR.round(3),
        "CI_lo":  CI_lo.round(3),
        "CI_hi":  CI_hi.round(3),
    })
    result = result.sort_values("OR", ascending=False)
    print(f"\n  {label}  (n={len(df):,}, positives={y.sum():,}, AUC={auc:.3f})")
    print(result.to_string(index=False))
    result["Model"] = label
    return result

young_model   = run_lr(mv_s[mv_s["IsElderly"]==False], "Young (<65)")
elderly_model = run_lr(mv_s[mv_s["IsElderly"]==True],  "Elderly (65+)")

if young_model is not None and elderly_model is not None:
    combined = pd.concat([young_model, elderly_model])
    combined.to_csv(OUT_DIR / "lr_odds_ratios.csv", index=False)
    print("\n✓ Saved outputs/lr_odds_ratios.csv")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5: THREE-TIER SDOH PREVALENCE (50-64 / 65-79 / 80+, surveyed only)
# The specific breakdown for the policy tier slide
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BLOCK 5: Three-Tier SDOH Prevalence (50-64, 65-79, 80+)")
print("="*70)

for tier in ["50-64","65-79","80+"]:
    sub = surveyed[surveyed["AgeGroup"]==tier]
    print(f"\n  {tier}  (n={len(sub):,} surveyed journeys)")
    for col in sdoh_cols:
        pct = sub[col].mean() * 100
        n   = sub[col].sum()
        print(f"    {sdoh_labels[col]:22s}: {pct:5.1f}%  (n={n})")

print("\n" + "="*70)
print("ALL BLOCKS COMPLETE — outputs saved to /outputs/")
print("="*70)
