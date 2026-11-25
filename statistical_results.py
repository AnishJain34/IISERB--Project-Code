"""
Comprehensive Window-Level Statistical Testing Pipeline (Tables 4, 5, 6, 7)
- Computes statistics across ALL WINDOWS without collapsing to participant medians.
- Uses 40 EDA features and 84 PPG features (124 features total).
- Generates Table 4, Table 5, Table 6, and Table 7 in both CSV and LaTeX formats.
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================ PATHS ============================
DATA_DIR = Path(r"C:\Users\Admin\Downloads\Step1")
SUDS_FILE = Path(r"C:\Users\Admin\Downloads\Selected_Participants_Combined_Phase_Features\participants_145_self_report.csv")
OUT_DIR = Path(r"D:\automate\window_level_statistical_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "baseline": DATA_DIR / "Baseline.csv",
    "2back": DATA_DIR / "2backpart12.csv",
    "speechrest": DATA_DIR / "SpeechRest.csv",
}

# ============================ HELPERS ============================
def clean_pid(value):
    text = str(value).strip().upper()
    m = re.fullmatch(r"P0*(\d+)", text)
    return f"P{int(m.group(1))}" if m else text

def get_modality(feature_name):
    name = feature_name.lower()
    if any(k in name for k in ["hrv",  "ppi"]):
        return "PPG"
    if any(k in name for k in ["scr_", "sc_", "scl", "phasic", "eda_"]):
        return "EDA"
    return "Other"

def apply_bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if valid.sum() > 0:
        q[valid] = multipletests(p[valid], method="fdr_bh")[1]
    return q

# ============================ TABLE 4: WINDOW-LEVEL MEDIANS ============================
def generate_table_4(baseline_df, twoback_df, speechrest_df):
    print("-> Calculating Table 4: Window-level medians for SCL and Pulse Rate...")
    
    # Locate SCL and Pulse rate columns
    scl_cols = [c for c in baseline_df.columns if "scl_mean" in c.lower() or c.lower().endswith("scl")]
    hr_cols = [c for c in baseline_df.columns if "hr_mean" in c.lower() or "pulse_rate" in c.lower() or "heart_rate" in c.lower()]
    
    scl_col = scl_cols[0] if scl_cols else "EDA_SCL_mean"
    hr_col = hr_cols[0] if hr_cols else "PPG_HR_mean"

    b_scl = float(pd.to_numeric(baseline_df[scl_col], errors="coerce").dropna().median())
    two_scl = float(pd.to_numeric(twoback_df[scl_col], errors="coerce").dropna().median())
    sr_scl = float(pd.to_numeric(speechrest_df[scl_col], errors="coerce").dropna().median())

    b_hr = float(pd.to_numeric(baseline_df[hr_col], errors="coerce").dropna().median())
    two_hr = float(pd.to_numeric(twoback_df[hr_col], errors="coerce").dropna().median())
    sr_hr = float(pd.to_numeric(speechrest_df[hr_col], errors="coerce").dropna().median())

    t4 = pd.DataFrame({
        "Measure": ["SCL (μS)", "Pulse rate (bpm)"],
        "Baseline": [round(b_scl, 2), round(b_hr, 1)],
        "2-Back": [round(two_scl, 2), round(two_hr, 1)],
        "Speech/Rest": [round(sr_scl, 2), round(sr_hr, 1)]
    })
    
    t4.to_csv(OUT_DIR / "Table_4_window_medians.csv", index=False)
    return t4

# ============================ TABLE 5: SUDS WILCOXON TESTS ============================
def generate_table_5(suds_df):
    print("-> Calculating Table 5: Within-subject SUDS Wilcoxon tests...")
    df = suds_df.copy()
    df["Participant"] = df["Participant"].map(clean_pid)

    comparisons = [
        ("All", "Baseline vs. after 2-back", "SUDS_beforeExp", "SUDS_afterExp", None),
        ("All", "after 2-back vs. after activity", "SUDS_afterExp", "SUDS_afterAct", None),
        ("All", "Baseline vs. after activity", "SUDS_beforeExp", "SUDS_afterAct", None),
        ("Stressed", "Baseline vs. after 2-back", "SUDS_beforeExp", "SUDS_afterExp", "Stressed"),
        ("Stressed", "Baseline vs. after activity", "SUDS_beforeExp", "SUDS_afterAct", "Stressed"),
        ("Control", "Baseline vs. after 2-back", "SUDS_beforeExp", "SUDS_afterExp", "Control"),
        ("Control", "Baseline vs. after activity", "SUDS_beforeExp", "SUDS_afterAct", "Control"),
    ]

    rows = []
    for grp, comp_name, col1, col2, cond in comparisons:
        sub = df if cond is None else df[df["Condition"].str.lower() == cond.lower()]
        paired = sub[[col1, col2]].dropna()
        x = pd.to_numeric(paired[col1], errors="coerce")
        y = pd.to_numeric(paired[col2], errors="coerce")
        diff = y - x
        nonzero = diff[diff != 0]
        
        w_stat, p_val = stats.wilcoxon(nonzero, alternative="two-sided")
        z = stats.norm.ppf(1.0 - p_val / 2.0) if p_val < 1.0 else 0.0
        r = float(np.sign(diff.median()) * abs(z) / np.sqrt(len(paired)))

        rows.append({
            "Group": grp,
            "Comparison": comp_name,
            "n": len(paired),
            "p": "<.001" if p_val < 0.001 else f"{p_val:.3f}",
            "r": f"{r:.2f}" if r >= 0 else f"-{abs(r):.2f}"
        })
    
    t5 = pd.DataFrame(rows)
    t5.to_csv(OUT_DIR / "Table_5_SUDS_Wilcoxon.csv", index=False)
    return t5

# ============================ TABLE 6 & 7: 40 EDA & 84 PPG FEATURES ============================
def generate_tables_6_and_7(baseline_df, twoback_df, speechrest_df):
    print("-> Calculating Table 6 & 7: Window-level tests on 40 EDA and 84 PPG features...")

    # Identify clean feature columns
    meta = {"participant", "phase", "condition", "source", "file", "window", "timestamp", "status"}
    common_cols = [c for c in baseline_df.columns if not any(m in c.lower() for m in meta)]

    eda_cols = [c for c in common_cols if get_modality(c) == "EDA"][:40]
    ppg_cols = [c for c in common_cols if get_modality(c) == "PPG"][:84]
    
    print(f"   Using {len(eda_cols)} EDA features and {len(ppg_cols)} PPG features ({len(eda_cols) + len(ppg_cols)} total).")

    # Define phases
    phase_dict = {
        "2-Back (Whole)": twoback_df,
        "2-Back Part 1": twoback_df[twoback_df["Phase"].astype(str).str.contains("Part1", case=False)],
        "2-Back Part 2": twoback_df[twoback_df["Phase"].astype(str).str.contains("Part2", case=False)],
        "Speech (Stressed)": speechrest_df[speechrest_df["Condition"].astype(str).str.lower() == "stressed"],
        "Rest (Control)": speechrest_df[speechrest_df["Condition"].astype(str).str.lower() == "control"],
    }

    t6_rows = []
    all_feature_results = []

    for phase_name, p_df in phase_dict.items():
        if len(p_df) == 0:
            continue
        
        eda_records, ppg_records = [], []
        
        # Test EDA features
        for f in eda_cols:
            b_vals = pd.to_numeric(baseline_df[f], errors="coerce").dropna()
            p_vals = pd.to_numeric(p_df[f], errors="coerce").dropna()
            if len(b_vals) >= 5 and len(p_vals) >= 5:
                u_stat, p_val = stats.mannwhitneyu(p_vals, b_vals, alternative="two-sided")
                n1, n2 = len(p_vals), len(b_vals)
                z = (u_stat - (n1 * n2 / 2.0)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
                r = float(np.sign(p_vals.median() - b_vals.median()) * abs(z) / np.sqrt(n1 + n2))
                eda_records.append({"Phase": phase_name, "Modality": "EDA", "Feature": f, "p": p_val, "r": r, "median_b": b_vals.median()})

        # Test PPG features
        for f in ppg_cols:
            b_vals = pd.to_numeric(baseline_df[f], errors="coerce").dropna()
            p_vals = pd.to_numeric(p_df[f], errors="coerce").dropna()
            if len(b_vals) >= 5 and len(p_vals) >= 5:
                u_stat, p_val = stats.mannwhitneyu(p_vals, b_vals, alternative="two-sided")
                n1, n2 = len(p_vals), len(b_vals)
                z = (u_stat - (n1 * n2 / 2.0)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
                r = float(np.sign(p_vals.median() - b_vals.median()) * abs(z) / np.sqrt(n1 + n2))
                ppg_records.append({"Phase": phase_name, "Modality": "PPG", "Feature": f, "p": p_val, "r": r, "median_b": b_vals.median()})

        eda_res = pd.DataFrame(eda_records)
        ppg_res = pd.DataFrame(ppg_records)
        
        if not eda_res.empty:
            eda_res["q"] = apply_bh_fdr(eda_res["p"])
            all_feature_results.append(eda_res)
        if not ppg_res.empty:
            ppg_res["q"] = apply_bh_fdr(ppg_res["p"])
            all_feature_results.append(ppg_res)

        t6_rows.append({
            "Phase vs. Baseline": phase_name,
            "EDA p < .05": int((eda_res["p"] < 0.05).sum()) if not eda_res.empty else 0,
            "EDA FDR": int((eda_res["q"] < 0.05).sum()) if not eda_res.empty else 0,
            "PPG p < .05": int((ppg_res["p"] < 0.05).sum()) if not ppg_res.empty else 0,
            "PPG FDR": int((ppg_res["q"] < 0.05).sum()) if not ppg_res.empty else 0,
        })

    t6 = pd.DataFrame(t6_rows)
    t6.to_csv(OUT_DIR / "Table_6_feature_counts.csv", index=False)
    
    if all_feature_results:
        full_df = pd.concat(all_feature_results, ignore_index=True)
        full_df.to_csv(OUT_DIR / "All_Window_Level_Feature_Tests.csv", index=False)

    return t6


def compute_table_7():
    DATA_DIR = Path(r"D:\EDA_DATA\Selected_Participants_Combined_Phase_Features")
    ROSTER_FILE = Path(r"D:\automate\participants_145_self_report.csv")

    roster = pd.read_csv(ROSTER_FILE)
    roster["Participant"] = roster["Participant"].astype(str).str.strip().str.upper()
    cond_map = dict(zip(roster["Participant"], roster["Condition"].astype(str).str.strip().str.capitalize()))

    b = pd.read_csv(DATA_DIR / "Baseline.csv", low_memory=False)
    two = pd.read_csv(DATA_DIR / "2backpart.csv", low_memory=False)
    sr = pd.read_csv(DATA_DIR / "SpeechRest.csv", low_memory=False)

    sr["Participant"] = sr["Participant"].astype(str).str.strip().str.upper()
    sr["Condition"] = sr["Participant"].map(cond_map)
    speech = sr[sr["Condition"] == "Stressed"]
    rest = sr[sr["Condition"] == "Control"]

    features = [
        ("SCL mean (\textmu S)", "EDA_SCL_mean"),
        ("SCR count", "EDA_SCR_Peak_count"),
        ("SCR amplitude (mean)", "EDA_SCR_Amplitude_mean"),
        ("SC std", "EDA_SC_std"),
        ("Pulse rate (bpm)", "PPG_HRV_HR_mean"),
        ("RMSSD (ms)", "PPG_HRV_HRV_RMSSD"),
        ("SDNN (ms)", "PPG_HRV_HRV_SDNN"),
        ("LF/HF", "PPG_HRV_HRV_LFHF"),
        ("PPG amplitude (mean)", "PPG_HRV_PPG_Amplitude_mean")
    ]

    rows = []
    for label, col in features:
        b_vals = pd.to_numeric(b[col], errors="coerce").dropna()
        b_med = float(b_vals.median())
        row = {"Feature": label, "Baseline (median)": round(b_med, 2) if b_med > 1 else round(b_med, 3)}

        for ph_name, ph_df in [("2-Back", two), ("Speech", speech), ("Rest", rest)]:
            p_vals = pd.to_numeric(ph_df[col], errors="coerce").dropna()
            p_med = float(p_vals.median())
            delta = p_med - b_med

            # Non-parametric Mann-Whitney across all windows
            u_stat, p_val = stats.mannwhitneyu(p_vals, b_vals, alternative="two-sided")
            n1, n2 = len(p_vals), len(b_vals)
            z = (u_stat - (n1 * n2 / 2.0)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
            r = float(np.sign(delta) * abs(z) / np.sqrt(n1 + n2))

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            row[f"{ph_name} r"] = f"{r:.2f}{sig}" if r >= 0 else f"-{abs(r):.2f}{sig}"

        rows.append(row)

    t7 = pd.DataFrame(rows)
    t7.to_csv(OUT_DIR / "Table_7_Wilcoxon_effect_size.csv", index=False)
    return t7


# ============================ MAIN ============================
def main():
    print("\n" + "=" * 70)
    print("  RUNNING WINDOW-LEVEL STATISTICAL TESTING PIPELINE (TABLES 4 - 7)")
    print("=" * 70)

    # 1. Load Datasets
    print("\n[Step 1] Loading data files...")
    suds_df = pd.read_csv(SUDS_FILE, low_memory=False)
    baseline_df = pd.read_csv(FEATURE_FILES["baseline"], low_memory=False)
    twoback_df = pd.read_csv(FEATURE_FILES["2back"], low_memory=False)
    speechrest_df = pd.read_csv(FEATURE_FILES["speechrest"], low_memory=False)

    print(f"  * Roster: {len(suds_df)} participants")
    print(f"  * Baseline Windows: {len(baseline_df)}")
    print(f"  * 2-Back Windows: {len(twoback_df)}")
    print(f"  * Speech/Rest Windows: {len(speechrest_df)}")

    # 2. Compute Table 4
    print("\n[Step 2] Table 4: Phase-wise Window-Level Medians")
    t4 = generate_table_4(baseline_df, twoback_df, speechrest_df)
    print(t4.to_string(index=False))

    # 3. Compute Table 5
    print("\n[Step 3] Table 5: Within-Subject SUDS Wilcoxon Tests")
    t5 = generate_table_5(suds_df)
    print(t5.to_string(index=False))

    # 4. Compute Table 6 & 7
    print("\n[Step 4] Table 6: Feature Counts (40 EDA and 84 PPG Features)")
    t6 = generate_tables_6_and_7(baseline_df, twoback_df, speechrest_df)
    print(t6.to_string(index=False))
    t7 = compute_table_7()
    print(t7.to_string(index=False))
    print("\n" + "=" * 70)
    print(f"  SUCCESS: All outputs saved to: {OUT_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
