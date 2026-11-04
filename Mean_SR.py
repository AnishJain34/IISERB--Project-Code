# SELF-REPORT: 145 participants, one CSV
# Figure 1: Mean SIAS (NSA vs SA)
# Figure 2: Overall SUDS and DASS-21
# Figure 3: SA vs NSA SUDS and DASS-21


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Admin\Downloads")
CSV_PATH = BASE_DIR / "participants_145_self_report.csv"
OUTPUT_DIR = BASE_DIR / "SIAS_SUDS_DASS_145_OUTPUT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N, EXPECTED_SA, EXPECTED_NSA = 145, 52, 93
SHOW_FIGURES = True

BLUE = "#1f77b4"
NSA_COLOR = "#7fb2d9"
SA_COLOR = "#1f4e8c"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def mean_ci(series):
    """Mean, approximate 95% CI half-width, and valid n."""
    v = pd.to_numeric(series, errors="coerce").dropna()
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(v.mean())
    ci = float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, ci, n


from scipy.stats import t


def descriptive_stats(series):
    """Calculate N, mean, SD and two-sided 95% CI."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": np.nan, "sd": np.nan,
            "ci_low": np.nan, "ci_high": np.nan
        }

    mean = values.mean()
    sd = values.std(ddof=1) if n > 1 else np.nan

    if n > 1:
        se = sd / np.sqrt(n)
        critical = t.ppf(0.975, df=n - 1)
        ci_low = mean - critical * se
        ci_high = mean + critical * se
    else:
        ci_low = ci_high = np.nan

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high
    }


def save_latex_table(df):
    """Create publication-ready LaTeX table for SUDS and DASS-21."""

    measures = [
        ("SUDS after Baseline", "SUDS_beforeExp"),
        ("SUDS after 2-Back", "SUDS_afterExp"),
        ("SUDS after Speech/Rest", "SUDS_afterAct"),
        ("DASS-21 Depression", "Depression_Score"),
        ("DASS-21 Anxiety", "Anxiety_Score"),
        ("DASS-21 Stress", "Stress_Score"),
    ]

    rows = []

    for label, column in measures:
        stats = descriptive_stats(df[column])

        rows.append({
            "Measure": label,
            "n": stats["n"],
            "Mean": stats["mean"],
            "SD": stats["sd"],
            "95% CI": (
                f"[{stats['ci_low']:.2f}, "
                f"{stats['ci_high']:.2f}]"
                if stats["n"] > 1 else "--"
            )
        })

    summary = pd.DataFrame(rows)

    # Save complete numerical results
    summary.to_csv(
        OUTPUT_DIR / "SUDS_DASS_Descriptive_Statistics.csv",
        index=False
    )

    # Generate LaTeX table
    latex_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Measure} & \textbf{$n$} & "
        r"\textbf{Mean} & \textbf{SD} & "
        r"\textbf{95\% CI} \\",
        r"\midrule"
    ]

    for _, row in summary.iterrows():
        mean = f"{row['Mean']:.1f}" if pd.notna(row["Mean"]) else "--"
        sd = f"{row['SD']:.2f}" if pd.notna(row["SD"]) else "--"

        latex_lines.append(
            f"{row['Measure']} & "
            f"{row['n']} & "
            f"{mean} & "
            f"{sd} & "
            f"{row['95% CI']} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])

    latex_path = OUTPUT_DIR / "SUDS_DASS_Descriptive_Table.tex"

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print("\nDescriptive Statistics:")
    print(summary.to_string(index=False))

    print(f"\nSaved CSV: {OUTPUT_DIR / 'SUDS_DASS_Descriptive_Statistics.csv'}")
    print(f"Saved LaTeX: {latex_path}")

    return summary

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def add_labels(ax, bars, means, cis, ns, offset=0.8):
    for bar, mean, ci, n in zip(bars, means, cis, ns):
        if pd.notna(mean) and n:
            ci = 0 if pd.isna(ci) else ci
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + ci + offset,
                f"{mean:.1f}\n(n={n})",
                ha="center", va="bottom", fontsize=9
            )


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Place participants_145_self_report.csv in Downloads "
            "or change BASE_DIR."
        )

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

    if df["Participant"].duplicated().any():
        dup = df.loc[df["Participant"].duplicated(False), "Participant"].tolist()
        raise ValueError(f"Duplicate participant IDs: {dup}")

    if len(df) != EXPECTED_N:
        raise ValueError(f"Expected {EXPECTED_N} rows; found {len(df)}.")

    if set(df["SA_NSA"].dropna().unique()) - {"SA", "NSA"}:
        raise ValueError("SA_NSA must contain only SA and NSA.")

    n_sa = int((df["SA_NSA"] == "SA").sum())
    n_nsa = int((df["SA_NSA"] == "NSA").sum())
    if (n_sa, n_nsa) != (EXPECTED_SA, EXPECTED_NSA):
        raise ValueError(
            f"Expected SA={EXPECTED_SA}, NSA={EXPECTED_NSA}; "
            f"found SA={n_sa}, NSA={n_nsa}."
        )

    invalid_conditions = set(df["Condition"].dropna().unique()) - {"Control", "Stressed"}
    if invalid_conditions:
        raise ValueError(f"Unexpected Condition labels: {invalid_conditions}")

    score_cols = [
        "SIAS", "SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct",
        "Depression_Score", "Anxiety_Score", "Stress_Score"
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]:
        bad = df[col].notna() & ((df[col] < 0) | (df[col] > 100))
        if bad.any():
            raise ValueError(f"{col} contains values outside 0-100.")

    df.to_csv(OUTPUT_DIR / "participant_data_audit.csv", index=False)

    print(f"Participants: {len(df)} | NSA: {n_nsa} | SA: {n_sa}")
    print("Valid score counts:")
    for col in score_cols:
        print(f"  {col}: {df[col].notna().sum()}")
    return df


def figure1_sias(df):
    # Layout matching example 1: NSA first, SA second; mean SIAS + 95% CI.
    groups = [
        ("NSA", "Non-socially anxious\n(NSA)"),
        ("SA", "Socially anxious\n(SA)")
    ]
    means, cis, ns = [], [], []
    for group, _ in groups:
        m, ci, n = mean_ci(df.loc[df.SA_NSA == group, "SIAS"])
        means.append(m); cis.append(ci); ns.append(n)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=300)
    x = np.arange(2)
    bars = ax.bar(
        x, means, width=0.58, color=BLUE, yerr=cis,
        capsize=6, ecolor="#333333", zorder=3
    )
    add_labels(ax, bars, means, cis, ns, offset=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in groups], fontsize=10)
    ax.set_ylabel("Mean SIAS score (0-80)")
    ax.set_title("SIAS: Group classification (pre-study)", fontsize=13, pad=12)
    upper = max(m + c for m, c in zip(means, cis) if pd.notna(m)) + 10
    ax.set_ylim(0, max(80, upper))
    style_axis(ax)
    fig.text(0.02, 0.015, "Error bars: 95% confidence intervals.", fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    path = OUTPUT_DIR / "fig1_sias_mean_sa_vs_nsa_145.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def figure2_overall(df):
    # Layout matching example 2: overall SUDS on left, overall DASS on right.
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=300)

    suds_cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    suds_labels = [
        "After Baseline\n(before 2-Back)",
        "After 2-Back\n(before activity)",
        "After Activity\n(after Speech/Rest)"
    ]
    sm, sc, sn = [], [], []
    for col in suds_cols:
        m, ci, n = mean_ci(df[col]); sm.append(m); sc.append(ci); sn.append(n)
    x = np.arange(3)
    bars = ax_s.bar(x, sm, width=0.62, color=BLUE, yerr=sc, capsize=6, zorder=3)
    add_labels(ax_s, bars, sm, sc, sn, offset=1.0)
    ax_s.set_xticks(x); ax_s.set_xticklabels(suds_labels, fontsize=9)
    ax_s.set_ylabel("Mean SUDS score (0-100)")
    ax_s.set_title("SUDS: Three in-study assessment points", fontsize=11)
    top = max(m + c for m, c in zip(sm, sc) if pd.notna(m)) + 8
    ax_s.set_ylim(0, max(100, top))
    style_axis(ax_s)

    dass_cols = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
    dass_labels = ["Depression", "Anxiety", "Stress"]
    dm, dc, dn = [], [], []
    for col in dass_cols:
        m, ci, n = mean_ci(df[col]); dm.append(m); dc.append(ci); dn.append(n)
    bars = ax_d.bar(x, dm, width=0.62, color=BLUE, yerr=dc, capsize=6, zorder=3)
    add_labels(ax_d, bars, dm, dc, dn, offset=0.5)
    ax_d.set_xticks(x); ax_d.set_xticklabels(dass_labels, fontsize=9)
    ax_d.set_ylabel("Mean DASS-21 subscale score")
    ax_d.set_title("DASS-21: Post-study questionnaire", fontsize=11)
    top = max(m + c for m, c in zip(dm, dc) if pd.notna(m)) + 3
    ax_d.set_ylim(0, max(20, top))
    style_axis(ax_d)

    fig.suptitle("Self-report measures across the experimental protocol", fontsize=14, y=1.02)
    fig.text(
        0.02, 0.005,
        "Error bars: 95% confidence intervals. SUDS is measured during the "
        "experiment; DASS-21 is completed after all phases. Scores use "
        "different scales and are not directly comparable.",
        fontsize=8
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = OUTPUT_DIR / "fig2_overall_suds_dass_145.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def figure3_group_comparison(df):
    # Layout matching example 3: NSA and SA bars for SUDS and DASS.
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)
    groups = ["NSA", "SA"]
    colors = [NSA_COLOR, SA_COLOR]
    group_n = {g: int((df.SA_NSA == g).sum()) for g in groups}
    width = 0.34
    x = np.arange(3)

    suds_cols = ["SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct"]
    suds_labels = [
        "After Baseline\n(before 2-Back)",
        "After 2-Back\n(before activity)",
        "After Activity\n(after Speech/Rest)"
    ]
    for i, group in enumerate(groups):
        means, cis = [], []
        for col in suds_cols:
            m, ci, _ = mean_ci(df.loc[df.SA_NSA == group, col])
            means.append(m); cis.append(ci)
        ax_s.bar(
            x + (i - 0.5) * width, means, width,
            yerr=cis, capsize=5, color=colors[i], edgecolor="black",
            linewidth=0.5, label=f"{group} (n={group_n[group]})", zorder=3
        )
    ax_s.set_xticks(x); ax_s.set_xticklabels(suds_labels, fontsize=9)
    ax_s.set_ylabel("Mean SUDS score (0-100)")
    ax_s.set_title("SUDS: in-study assessment points", fontsize=11)
    ax_s.set_ylim(0, 100)
    ax_s.legend(frameon=False)
    style_axis(ax_s)

    dass_cols = ["Depression_Score", "Anxiety_Score", "Stress_Score"]
    dass_labels = ["Depression", "Anxiety", "Stress"]
    dass_max = 0
    for i, group in enumerate(groups):
        means, cis = [], []
        for col in dass_cols:
            m, ci, _ = mean_ci(df.loc[df.SA_NSA == group, col])
            means.append(m); cis.append(ci)
            if pd.notna(m):
                dass_max = max(dass_max, m + ci)
        ax_d.bar(
            x + (i - 0.5) * width, means, width,
            yerr=cis, capsize=5, color=colors[i], edgecolor="black",
            linewidth=0.5, label=f"{group} (n={group_n[group]})", zorder=3
        )
    ax_d.set_xticks(x); ax_d.set_xticklabels(dass_labels, fontsize=9)
    ax_d.set_ylabel("Mean DASS-21 subscale score")
    ax_d.set_title("DASS-21: post-study questionnaire", fontsize=11)
    ax_d.set_ylim(0, max(20, dass_max + 3))
    ax_d.legend(frameon=False)
    style_axis(ax_d)

    fig.suptitle(
        "Mean self-reported scores with 95% confidence intervals: "
        "socially anxious (SA) vs non-socially anxious (NSA)",
        fontsize=13, y=1.02
    )
    fig.text(
        0.02, 0.005,
        "Error bars: 95% confidence intervals. Group labels use the SA/NSA "
        "classification in the CSV. Missing scores are excluded separately "
        "for each measure. SUDS and DASS-21 use different scales.",
        fontsize=8
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = OUTPUT_DIR / "fig3_sa_nsa_suds_dass_145.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def save_summary(df):
    variables = [
        "SIAS", "SUDS_beforeExp", "SUDS_afterExp", "SUDS_afterAct",
        "Depression_Score", "Anxiety_Score", "Stress_Score"
    ]
    rows = []
    for variable in variables:
        for group in ["NSA", "SA", "Overall"]:
            values = df[variable] if group == "Overall" else df.loc[df.SA_NSA == group, variable]
            m, ci, n = mean_ci(values)
            clean = pd.to_numeric(values, errors="coerce").dropna()
            rows.append({
                "Variable": variable, "Group": group, "N_valid": n,
                "Mean": m, "SD": clean.std(ddof=1) if n > 1 else np.nan,
                "CI95_half_width": ci
            })
    summary = pd.DataFrame(rows)
    path = OUTPUT_DIR / "self_report_summary_statistics.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")
    return summary


def figure4_baseline_speech_rest(df):
    """
    Panel A: Baseline vs Speech (Stressed condition)
    Panel B: Baseline vs Rest (Control condition)
    Mean SUDS with 95% CI
    """

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5.5), dpi=300
    )

    panels = [
        (ax1, "Stressed", "Speech Activity"),
        (ax2, "Control", "Rest Activity")
    ]

    for ax, condition, activity in panels:

        sub = df[df["Condition"] == condition].copy()

        # Keep participants with both measurements
        sub = sub.dropna(
            subset=["SUDS_beforeExp", "SUDS_afterAct"]
        )

        baseline_mean, baseline_ci, n1 = mean_ci(
            sub["SUDS_beforeExp"]
        )

        activity_mean, activity_ci, n2 = mean_ci(
            sub["SUDS_afterAct"]
        )

        means = [baseline_mean, activity_mean]
        cis = [baseline_ci, activity_ci]
        ns = [n1, n2]

        x = np.arange(2)

        bars = ax.bar(
            x, means,
            width=0.55,
            color=[NSA_COLOR, SA_COLOR],
            yerr=cis,
            capsize=6,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

        add_labels(ax, bars, means, cis, ns, offset=1)

        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Baseline", activity],
            fontsize=10
        )

        ax.set_ylabel("Mean SUDS score (0–100)")
        ax.set_title(
            f"{activity} (n={len(sub)})",
            fontsize=12
        )

        ax.set_ylim(0, 100)
        style_axis(ax)

    fig.suptitle(
        "Self-reported distress: Baseline vs Speech/Rest",
        fontsize=14
    )

    fig.text(
        0.02, 0.01,
        "Error bars represent 95% confidence intervals. "
        "Speech corresponds to Stressed condition; "
        "Rest corresponds to Control condition.",
        fontsize=8
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    path = OUTPUT_DIR / "fig4_baseline_speech_rest_suds.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")

    print(f"Saved: {path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)

def main():
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")

    df = load_data()

    figure1_sias(df)
    figure2_overall(df)
    figure3_group_comparison(df)
    figure4_baseline_speech_rest(df)
    summary = save_summary(df)

    # NEW: SD + 95% CI + publication-ready LaTeX table
    descriptive_table = save_latex_table(df)

    print("\nCompleted.")
    print("Check the output folder for figures, CSV reports, and LaTeX table.")


if __name__ == "__main__":
    main()
