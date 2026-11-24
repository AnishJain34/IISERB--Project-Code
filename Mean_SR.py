# ==============================================================================
# SELF-REPORT ANALYSIS: 145 PARTICIPANTS (SIAS, SUDS, DASS-21)
#
# Figures Generated:
#   Figure 1: Mean SIAS (NSA vs SA)
#   Figure 2: Overall SUDS and DASS-21 Subscales
#   Figure 3: SA vs NSA Comparison on SUDS and DASS-21
#   Figure 4: Stressed Group SUDS Across Experimental Phases
#   Figure 5: Control Group SUDS Across Experimental Phases
#   Figure 6: Comparison of Stressed vs. Control Across Phases
#
# Robustness Guarantees:
#   - Any negative score (< 0) is treated as missing (NaN).
#   - Confidence intervals and error bars are bounded at zero (no negative limits).
#   - Valid n is tracked dynamically per scale (handles partial questionnaire completion).
# ==============================================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

# ------------------------------- PATH CONFIG ----------------------------------

BASE_DIR = Path(r"C:\Users\Admin\Downloads")
CSV_PATH = BASE_DIR / "Selected_Participants_Combined_Phase_Features" / "participants_145_self_report.csv"


OUTPUT_DIR = BASE_DIR / "SIAS_SUDS_DASS_145_OUTPUT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N, EXPECTED_SA, EXPECTED_NSA = 145, 52, 93
SHOW_FIGURES = False  # Set to True if running interactively in Jupyter/Spyder

# Colors
BLUE = "#1f77b4"
NSA_COLOR = "#7fb2d9"
SA_COLOR = "#1f4e8c"
STRESSED_COLOR = "#D55E00"
CONTROL_COLOR = "#0072B2"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


# --------------------------- STATISTICAL HELPERS ------------------------------

def mean_ci(series):
    """
    Computes mean, 95% CI half-width, and valid sample count.
    Treats any negative values as missing (NaN).
    Ensures CI lower bound does not drop below zero.
    """
    v = pd.to_numeric(series, errors="coerce")
    v = v[v.notna() & (v >= 0)]  # Non-negative values only
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(v.mean())
    ci = float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, ci, n


def descriptive_stats(series):
    """
    Computes N, Mean, SD, and two-sided 95% CI.
    Clamps lower CI to zero (scores cannot be negative).
    """
    values = pd.to_numeric(series, errors="coerce")
    values = values[values.notna() & (values >= 0)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0

    if n > 1:
        se = sd / np.sqrt(n)
        critical = float(t.ppf(0.975, df=n - 1))
        ci_low = max(0.0, mean - critical * se)  # Bound at zero (no negative limits)
        ci_high = mean + critical * se
    else:
        ci_low = ci_high = mean

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def add_labels(ax, bars, means, cis, ns, offset=1.0):
    for bar, mean, ci, n in zip(bars, means, cis, ns):
        if pd.notna(mean) and n > 0:
            err = 0.0 if pd.isna(ci) else ci
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + err + offset,
                f"{mean:.1f}\n(n={n})",
                ha="center", va="bottom", fontsize=9
            )


# ---------------------------- DATA LOADING ------------------------------------

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Self-report CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required = [
        "Participant", "SA_NSA", "Condition", "SIAS",
        "SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct",
        "Depression_Score", "Anxiety_Score", "Stress_Score"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["Participant"] = df["Participant"].astype(str).str.strip().str.upper()
    df["SA_NSA"] = df["SA_NSA"].astype(str).str.strip().str.upper()
    df["Condition"] = df["Condition"].astype(str).str.strip().str.capitalize()

    # Sanitize numerical scores: treat negative numbers as missing (NaN)
    score_cols = [
        "SIAS", "SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct",
        "Depression_Score", "Anxiety_Score", "Stress_Score"
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    # Validate SUDS score bounds (0–100)
    for col in ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]:
        df.loc[df[col] > 100, col] = np.nan

    n_sa = int((df["SA_NSA"] == "SA").sum())
    n_nsa = int((df["SA_NSA"] == "NSA").sum())
    print(f"Loaded {len(df)} participants | NSA: {n_nsa} | SA: {n_sa}")
    print("Valid score counts (non-negative):")
    for col in score_cols:
        print(f"  {col:<20}: {df[col].notna().sum()} valid")

    df.to_csv(OUTPUT_DIR / "participant_data_clean_audit.csv", index=False)
    return df


# ---------------------------- VISUALIZATIONS ----------------------------------

def figure1_sias(df):
    """Figure 1: Mean SIAS Score (NSA vs. SA)."""
    groups = [("NSA", "Non-socially anxious\n(NSA)"), ("SA", "Socially anxious\n(SA)")]
    means, cis, ns = [], [], []
    for group, _ in groups:
        m, ci, n = mean_ci(df.loc[df.SA_NSA == group, "SIAS"])
        means.append(m); cis.append(ci); ns.append(n)

    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=300)
    x = np.arange(2)
    # Clip error bars so they do not extend below 0
    yerr = [np.minimum(means, cis), cis]
    bars = ax.bar(x, means, width=0.55, color=BLUE, yerr=yerr, capsize=6, ecolor="#333333", zorder=3)
    add_labels(ax, bars, means, cis, ns, offset=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in groups], fontsize=10)
    ax.set_ylabel("Mean SIAS score (0–80)", fontsize=11)
    ax.set_title("SIAS: Pre-Study Social Anxiety Classification", fontsize=12, pad=12)
    upper = max(m + c for m, c in zip(means, cis) if pd.notna(m)) + 12
    ax.set_ylim(0, max(80, upper))
    style_axis(ax)

    fig.text(0.02, 0.015, "Error bars: 95% confidence intervals (bounded at 0).", fontsize=8)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    path = OUTPUT_DIR / "fig1_sias_mean_sa_vs_nsa.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path.name}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def figure2_overall(df):
    """Figure 2: Overall SUDS (In-Study) and DASS-21 (Post-Study)."""
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13.0, 5.6), dpi=300)

    # Left: SUDS
    suds_cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    suds_labels = ["After Baseline\n(Resting)", "After 2-Back\n(Cognitive Load)", "After Speech/Rest\n(Recovery)"]
    sm, sc, sn = [], [], []
    for col in suds_cols:
        m, ci, n = mean_ci(df[col]); sm.append(m); sc.append(ci); sn.append(n)

    x = np.arange(3)
    bars_s = ax_s.bar(x, sm, width=0.60, color=BLUE, yerr=[np.minimum(sm, sc), sc], capsize=5, zorder=3)
    add_labels(ax_s, bars_s, sm, sc, sn, offset=1.0)
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(suds_labels, fontsize=9)
    ax_s.set_ylabel("Mean SUDS Score (0–100)", fontsize=10)
    ax_s.set_title("Subjective Units of Distress (SUDS) Across Phases", fontsize=11)
    top_s = max(m + c for m, c in zip(sm, sc) if pd.notna(m)) + 10
    ax_s.set_ylim(0, max(100, top_s))
    style_axis(ax_s)

    # Right: DASS-21 Subscales
    dass_cols = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
    dass_labels = ["Depression", "Anxiety", "Stress"]
    dm, dc, dn = [], [], []
    for col in dass_cols:
        m, ci, n = mean_ci(df[col]); dm.append(m); dc.append(ci); dn.append(n)

    bars_d = ax_d.bar(x, dm, width=0.60, color=BLUE, yerr=[np.minimum(dm, dc), dc], capsize=5, zorder=3)
    add_labels(ax_d, bars_d, dm, dc, dn, offset=0.6)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(dass_labels, fontsize=9)
    ax_d.set_ylabel("Mean DASS-21 Subscale Score (0–42)", fontsize=10)
    ax_d.set_title("DASS-21 Subscale Profiles", fontsize=11)
    top_d = max(m + c for m, c in zip(dm, dc) if pd.notna(m)) + 4
    ax_d.set_ylim(0, max(25, top_d))
    style_axis(ax_d)

    fig.suptitle("Overall Self-Report Measures Across the Experimental Protocol", fontsize=13, y=1.01)
    fig.text(0.02, 0.005, "Error bars: 95% confidence intervals (bounded at 0). Missing values excluded per measure.", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = OUTPUT_DIR / "fig2_overall_suds_dass.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path.name}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def figure3_group_comparison(df):
    """Figure 3: SA vs. NSA Comparison on SUDS and DASS-21."""
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13.0, 5.8), dpi=300)
    groups = ["NSA", "SA"]
    colors = [NSA_COLOR, SA_COLOR]
    group_n = {g: int((df.SA_NSA == g).sum()) for g in groups}
    width = 0.35
    x = np.arange(3)

    # SUDS
    suds_cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    suds_labels = ["After Baseline\n(Resting)", "After 2-Back\n(Cognitive Load)", "After Speech/Rest\n(Recovery)"]
    for i, group in enumerate(groups):
        means, cis, ns = [], [], []
        for col in suds_cols:
            m, ci, n = mean_ci(df.loc[df.SA_NSA == group, col])
            means.append(m); cis.append(ci); ns.append(n)
        yerr = [np.minimum(means, cis), cis]
        ax_s.bar(
            x + (i - 0.5) * width, means, width, yerr=yerr, capsize=5,
            color=colors[i], edgecolor="black", linewidth=0.5,
            label=f"{group} (n={group_n[group]})", zorder=3
        )
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(suds_labels, fontsize=9)
    ax_s.set_ylabel("Mean SUDS Score (0–100)", fontsize=10)
    ax_s.set_title("SUDS: In-Study Assessment Points", fontsize=11)
    ax_s.set_ylim(0, 100)
    ax_s.legend(frameon=False)
    style_axis(ax_s)

    # DASS-21
    dass_cols = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
    dass_labels = ["Depression", "Anxiety", "Stress"]
    dass_max = 0
    for i, group in enumerate(groups):
        means, cis, ns = [], [], []
        for col in dass_cols:
            m, ci, n = mean_ci(df.loc[df.SA_NSA == group, col])
            means.append(m); cis.append(ci); ns.append(n)
            if pd.notna(m):
                dass_max = max(dass_max, m + ci)
        yerr = [np.minimum(means, cis), cis]
        ax_d.bar(
            x + (i - 0.5) * width, means, width, yerr=yerr, capsize=5,
            color=colors[i], edgecolor="black", linewidth=0.5,
            label=f"{group} (n={group_n[group]})", zorder=3
        )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(dass_labels, fontsize=9)
    ax_d.set_ylabel("Mean DASS-21 Subscale Score", fontsize=10)
    ax_d.set_title("DASS-21: Post-Study Subscales", fontsize=11)
    ax_d.set_ylim(0, max(22, dass_max + 4))
    ax_d.legend(frameon=False)
    style_axis(ax_d)

    fig.suptitle("Self-Report Scores: Socially Anxious (SA) vs. Non-Socially Anxious (NSA)", fontsize=13, y=1.01)
    fig.text(0.02, 0.005, "Error bars: 95% confidence intervals (bounded at 0). Groups defined by baseline SIAS classification.", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = OUTPUT_DIR / "fig3_sa_nsa_suds_dass.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path.name}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def figure4_mean_sr_stressed(df):
    """Figure 4: Stressed Group SUDS Across Phases."""
    stressed = df[df["Condition"] == "Stressed"].copy()
    cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    labels = ["Baseline", "After 2-Back", "After Speech/Activity"]

    means = np.array([mean_ci(stressed[col])[0] for col in cols])
    cis = np.array([mean_ci(stressed[col])[1] for col in cols])
    yerr = [np.minimum(means, cis), cis]

    plt.figure(figsize=(7.5, 4.8), dpi=300)
    plt.errorbar(
        labels, means, yerr=yerr,
        marker="o", capsize=6, linewidth=2,
        color=STRESSED_COLOR, ecolor="#333333"
    )
    plt.ylabel("Mean Self-Reported Distress (SUDS)", fontsize=10)
    plt.xlabel("Experimental Phase", fontsize=10)
    plt.title(f"Distress Trajectory – Stressed Cohort (n={len(stressed)})", fontsize=12)
    plt.ylim(bottom=0, top=max(100, np.max(means + cis) + 8))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "fig4_mean_sr_stressed.png"
    plt.savefig(path, dpi=300)
    print(f"Saved: {path.name}")
    plt.close()


def figure5_mean_sr_rest(df):
    """Figure 5: Control/Rest Group SUDS Across Phases."""
    control = df[df["Condition"] == "Control"].copy()
    cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    labels = ["Baseline", "After 2-Back", "After Rest"]

    means = np.array([mean_ci(control[col])[0] for col in cols])
    cis = np.array([mean_ci(control[col])[1] for col in cols])
    yerr = [np.minimum(means, cis), cis]

    plt.figure(figsize=(7.5, 4.8), dpi=300)
    plt.errorbar(
        labels, means, yerr=yerr,
        marker="o", capsize=6, linewidth=2,
        color=CONTROL_COLOR, ecolor="#333333"
    )
    plt.ylabel("Mean Self-Reported Distress (SUDS)", fontsize=10)
    plt.xlabel("Experimental Phase", fontsize=10)
    plt.title(f"Distress Trajectory – Control/Rest Cohort (n={len(control)})", fontsize=12)
    plt.ylim(bottom=0, top=max(100, np.max(means + cis) + 8))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "fig5_mean_sr_rest.png"
    plt.savefig(path, dpi=300)
    print(f"Saved: {path.name}")
    plt.close()


def figure6_mean_sr_comparison(df):
    """Figure 6: Stressed vs. Control Group SUDS Comparison."""
    cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    labels = ["Baseline", "After 2-Back", "After Speech/Rest"]

    stressed = df[df["Condition"] == "Stressed"]
    control = df[df["Condition"] == "Control"]

    s_means = np.array([mean_ci(stressed[col])[0] for col in cols])
    s_cis = np.array([mean_ci(stressed[col])[1] for col in cols])
    c_means = np.array([mean_ci(control[col])[0] for col in cols])
    c_cis = np.array([mean_ci(control[col])[1] for col in cols])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8.5, 5.0), dpi=300)
    plt.bar(
        x - width/2, s_means, width, yerr=[np.minimum(s_means, s_cis), s_cis],
        capsize=5, label=f"Stressed (n={len(stressed)})", color=STRESSED_COLOR
    )
    plt.bar(
        x + width/2, c_means, width, yerr=[np.minimum(c_means, c_cis), c_cis],
        capsize=5, label=f"Control/Rest (n={len(control)})", color=CONTROL_COLOR
    )

    plt.xticks(x, labels, fontsize=10)
    plt.ylabel("Mean SUDS Score (0–100)", fontsize=10)
    plt.title("Subjective Distress Comparison: Stressed vs. Control", fontsize=12)
    plt.ylim(0, 100)
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "fig6_mean_sr_comparison.png"
    plt.savefig(path, dpi=300)
    print(f"Saved: {path.name}")
    plt.close()


# -------------------------- DESCRIPTIVE TABLES --------------------------------

def save_latex_table(df):
    """Generates publication-ready CSV and LaTeX table for SUDS and DASS-21."""
    measures = [
        ("SUDS Baseline", "SUDS_beforeExp"),
        ("SUDS 2-Back", "SUDS_afterExp"),
        ("SUDS Speech/Rest", "SUDS_afterAct"),
        ("DASS-21 Depression", "Depression_Score"),
        ("DASS-21 Anxiety", "Anxiety_Score"),
        ("DASS-21 Stress", "Stress_Score"),
        ("SIAS Score", "SIAS"),
    ]

    rows = []
    for label, col in measures:
        stats = descriptive_stats(df[col])
        rows.append({
            "Measure": label,
            "n": stats["n"],
            "Mean": stats["mean"],
            "SD": stats["sd"],
            "95% CI": f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]" if stats["n"] > 1 else "--"
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "SUDS_DASS_Descriptive_Statistics.csv", index=False)

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Descriptive statistics of subjective self-report measures ($N=145$).}",
        r"\label{tab:self_report_descriptives}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Measure} & \textbf{$n$} & \textbf{Mean} & \textbf{SD} & \textbf{95\% CI} \\",
        r"\midrule"
    ]
    for _, row in summary.iterrows():
        mean_str = f"{row['Mean']:.2f}" if pd.notna(row["Mean"]) else "--"
        sd_str = f"{row['SD']:.2f}" if pd.notna(row["SD"]) else "--"
        latex_lines.append(f"{row['Measure']} & {row['n']} & {mean_str} & {sd_str} & {row['95% CI']} \\\\")
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_path = OUTPUT_DIR / "SUDS_DASS_Descriptive_Table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print("\nDescriptive Summary Table:")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------- MAIN --------------------------------------

def main():
    print("=" * 70)
    print("  SELF-REPORT PROTOCOL ANALYSIS (NO NEGATIVES, ROBUST MISSING HANDLING)")
    print("=" * 70)
    print(f"Input  : {CSV_PATH}")
    print(f"Output : {OUTPUT_DIR}\n")

    df = load_data()

    figure1_sias(df)
    figure2_overall(df)
    figure3_group_comparison(df)
    figure4_mean_sr_stressed(df)
    figure5_mean_sr_rest(df)
    figure6_mean_sr_comparison(df)
    save_latex_table(df)

    print("\n" + "=" * 70)
    print(f"All 6 figures and LaTeX tables successfully generated in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()