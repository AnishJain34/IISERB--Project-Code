"""
Phase-wise summary of the raw Shimmer GSR and PPG recordings -> two LaTeX tables (GSR first, then PPG).

Layout follows the reference image:
  Table (GSR): Phase | Duration | SC (uS) | SkR (kOhm)
  Table (PPG): Phase | Duration | PPG (mV) | IBI (ms) | HR (bpm)
Rows: Baseline, 2-Back Task (Part 1 + Part 2 durations added), Speech/Rest (Stressed or Control file).



Outputs (OUTPUT_DIR): table_GSR.tex, table_PPG.tex, tables_raw_summary.tex (both),
                      table4_per_participant.csv, table4_GSR_summary.csv, table4_PPG_summary.csv, table4_missing_files.csv
"""
import os
import glob
import gc
import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================
ROOT = os.environ.get("RAW_ROOT", r"D:\automate\EDA_Phase")
FILE_TAG = "EDA"
OUTPUT_DIR = os.environ.get("RAW_OUT", "Table4_MAD_Format")
CHUNK_SIZE = 100_000
SHOW_N = False            # True adds a column with the number of participants per phase
VALUE_DECIMALS = 3        # as in the reference image (2.002 +/- 2.238)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. SELECTED PARTICIPANTS
# ============================================================
PARTICIPANTS = [
    "P7", "P9", "P10", "P11", "P12", "P16", "P20", "P21", "P25", "P26",
    "P27", "P29", "P30", "P31", "P34", "P39", "P41", "P43", "P44", "P46",
    "P47", "P48", "P50", "P51", "P52", "P53", "P55", "P57", "P58", "P59",
    "P61", "P63", "P65", "P67", "P68", "P69", "P70", "P71", "P73", "P76",
    "P77", "P79", "P80", "P81", "P82", "P83", "P84", "P85", "P87", "P88",
    "P89", "P90", "P91", "P92", "P93", "P95", "P96", "P98", "P99", "P100",
    "P101", "P102", "P104", "P105", "P106", "P108", "P109", "P110", "P111", "P112",
    "P114", "P115", "P116", "P117", "P118", "P119", "P121", "P122", "P123", "P125",
    "P130", "P132", "P133", "P134", "P135", "P136", "P137", "P138", "P139", "P140",
    "P141", "P142", "P143", "P144", "P145", "P146", "P147", "P148", "P149", "P150",
    "P151", "P152", "P153", "P155", "P158", "P159", "P160", "P161", "P164", "P165",
    "P166", "P167", "P207", "P209", "P211", "P213", "P214", "P226", "P229", "P231",
    "P232", "P233", "P234", "P235", "P236", "P237", "P238", "P239", "P240", "P242",
    "P243", "P244", "P245", "P246", "P247", "P248", "P249", "P250", "P251", "P254",
    "P255", "P258", "P259", "P261", "P263", "P264", "P266", "P267", "P268", "P269",
    "P270", "P271", "P272", "P273", "P275", "P276", "P277", "P278", "P279", "P280",
    "P281", "P283", "P285", "P286", "P287", "P288"
]
if os.environ.get("RAW_PARTICIPANTS"):            # optional override, e.g. RAW_PARTICIPANTS=P7,P9
    PARTICIPANTS = [p.strip() for p in os.environ["RAW_PARTICIPANTS"].split(",") if p.strip()]

# ============================================================
# 3. EXPERIMENTAL PHASES  (labels as in the reference image)
# ============================================================
PHASES = {
    "Baseline": {"suffixes": ["Baseline"], "label": "Baseline"},
    "2Back": {"suffixes": ["2BackPart1", "2BackPart2"], "label": "2-Back Task"},      # both parts, durations added
    "SpeechRest": {"suffixes": ["SpeechRest_Stressed", "SpeechRest_Control"], "label": "Speech/Rest"},
}
PHASE_ORDER = list(PHASES.keys())

# ============================================================
# 4. RAW SIGNAL DEFINITIONS
# ============================================================
SIGNALS = {
    "GSR": {
        "GSR_Skin_Conductance_CAL": {"label": "SC", "unit": r"$\mu$S"},
        "GSR_Skin_Resistance_CAL": {"label": "SkR", "unit": r"k$\Omega$"},
    },
    "PPG": {
        "PPG_A13_CAL": {"label": "PPG", "unit": "mV"},
        "PPG_IBI_CAL": {"label": "IBI", "unit": "ms"},
        "PPGtoHR_CAL": {"label": "HR", "unit": "bpm"},
    },
}
CAPTIONS = {
    "GSR": ("Summary of raw GSR recordings across experimental phases. Values are reported as mean $\\pm$ standard "
            "deviation (SD) across participants. SC: skin conductance; SkR: skin resistance."),
    "PPG": ("Summary of raw PPG recordings across experimental phases. Values are reported as mean $\\pm$ standard "
            "deviation (SD) across participants. IBI: inter-beat interval; HR: heart rate."),
}
LABELS = {"GSR": "tab:raw_gsr_summary", "PPG": "tab:raw_ppg_summary"}
ALL_SIGNAL_COLUMNS = [c for kit in ("PPG", "GSR") for c in SIGNALS[kit]]


# ============================================================
# 5. FIND RECORDING FILE
# ============================================================
def find_phase_file(folder, pid, suffix):
    files = glob.glob(os.path.join(folder, f"{pid}_{suffix}_{FILE_TAG}.csv"))
    files = [f for f in files if "_COMPLETE" not in os.path.basename(f) and "_Phase_Summary" not in os.path.basename(f)]
    return files[0] if files else None


# ============================================================
# 6. PROCESS LARGE CSV FILES
# ============================================================
def process_large_csv(filepaths):
    signal_sum = {s: 0.0 for s in ALL_SIGNAL_COLUMNS}
    signal_count = {s: 0 for s in ALL_SIGNAL_COLUMNS}
    total_samples, total_duration = 0, 0.0
    for filepath in filepaths:
        print("   Reading:", os.path.basename(filepath))
        header = pd.read_csv(filepath, nrows=0, encoding="utf-8-sig")
        timestamp_col = next((c for c in header.columns if c.endswith("Timestamp_FormattedUnix_CAL")), None)
        if timestamp_col is None:
            raise ValueError(f"Timestamp column missing: {filepath}")
        actual = {s: next((c for c in header.columns if c.endswith(s)), None) for s in ALL_SIGNAL_COLUMNS}
        usecols = [timestamp_col] + [c for c in actual.values() if c is not None]
        file_min = file_max = None
        for chunk in pd.read_csv(filepath, usecols=usecols, chunksize=CHUNK_SIZE, encoding="utf-8-sig"):
            total_samples += len(chunk)
            ts = pd.to_datetime(chunk[timestamp_col], format="%d-%m-%Y %H:%M:%S.%f", errors="coerce").dropna()
            if not ts.empty:
                file_min = ts.min() if file_min is None else min(file_min, ts.min())
                file_max = ts.max() if file_max is None else max(file_max, ts.max())
            for signal, column in actual.items():
                if column is None:
                    continue
                values = pd.to_numeric(chunk[column], errors="coerce")
                signal_sum[signal] += values.sum()
                signal_count[signal] += values.count()
            del chunk
        if file_min is not None and file_max is not None:
            duration = (file_max - file_min).total_seconds()
            if duration > 0:
                total_duration += duration            # Part 1 + Part 2 durations are added for the 2-Back task
        gc.collect()
    result = {"Duration_s": total_duration, "Samples": total_samples}
    for s in ALL_SIGNAL_COLUMNS:
        result[s] = signal_sum[s] / signal_count[s] if signal_count[s] > 0 else np.nan
    return result


# ============================================================
# 6b. FORMATTING: DURATION IN MINUTES AND SECONDS
# ============================================================
def fmt_dur(seconds):
    """119.9 -> '2m 0s', 759 -> '12m 39s'."""
    total = int(round(seconds))
    return f"{total // 60}m {total % 60}s"


def fmt_sd_dur(seconds):
    """SD in seconds ('5s'); written as minutes and seconds only when it reaches one minute."""
    total = int(round(seconds))
    return f"{total}s" if total < 60 else f"{total // 60}m {total % 60}s"


def fmt_duration(mean, sd):
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return f"{fmt_dur(mean)} (NA)"
    return f"{fmt_dur(mean)} $\\pm$ {fmt_sd_dur(sd)}"


def fmt_value(mean, sd, decimals=VALUE_DECIMALS):
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return f"{mean:.{decimals}f} (NA)"
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


# ============================================================
# 7. PROCESS PARTICIPANTS
# ============================================================
rows, missing = [], []
print("\n========== STARTING RAW SENSOR SUMMARY ==========")
print("Selected participants:", len(PARTICIPANTS))
for i, pid in enumerate(PARTICIPANTS, start=1):
    folder = os.path.join(ROOT, pid)
    print(f"\n[{i}/{len(PARTICIPANTS)}] Participant: {pid}")
    if not os.path.isdir(folder):
        print("   Participant folder missing")
        missing += [[pid, ph, "Participant folder missing"] for ph in PHASES]
        continue
    for phase_key, info in PHASES.items():
        filepaths = []
        for suffix in info["suffixes"]:
            fp = find_phase_file(folder, pid, suffix)
            if fp is None:
                # SpeechRest has two alternative names (Stressed / Control): only one of them exists per participant
                if phase_key != "SpeechRest":
                    missing.append([pid, phase_key, f"Missing file: {suffix}"])
            else:
                filepaths.append(fp)
        if phase_key == "SpeechRest" and not filepaths:
            missing.append([pid, phase_key, "Missing file: SpeechRest_Stressed / SpeechRest_Control"])
        if not filepaths:
            print(f"   {info['label']}: no recordings found")
            continue
        try:
            result = process_large_csv(filepaths)
            for kit in ("PPG", "GSR"):
                row = {"Participant": pid, "Kit": kit, "Phase": phase_key, "Duration_s": result["Duration_s"], "Samples": result["Samples"]}
                row.update({s: result[s] for s in SIGNALS[kit]})
                rows.append(row)
            print(f"   {info['label']} completed | Duration: {result['Duration_s']:.1f} s ({fmt_dur(result['Duration_s'])})")
        except Exception as error:
            missing.append([pid, phase_key, f"Processing error: {error}"])
            print(f"   ERROR: {pid} | {phase_key} | {error}")


# ============================================================
# 8. SAVE PARTICIPANT-LEVEL DATA
# ============================================================
P = pd.DataFrame(rows)
if P.empty:
    raise SystemExit("No recordings were processed - check ROOT, the participant list and the file names ({pid}_{phase}_EDA.csv).")
P.to_csv(os.path.join(OUTPUT_DIR, "table4_per_participant.csv"), index=False)
pd.DataFrame(missing, columns=["Participant", "Phase", "Reason"]).to_csv(os.path.join(OUTPUT_DIR, "table4_missing_files.csv"), index=False)
print("\nParticipant-level data saved.")


# ============================================================
# 10. BUILD ONE TABLE PER SENSOR KIT
# ============================================================
def summarise(kit):
    data = P[P["Kit"] == kit]
    cols = ["Duration_s"] + list(SIGNALS[kit])
    summary = data.groupby("Phase")[cols].agg(["mean", "std"]).reindex(PHASE_ORDER)
    counts = data.groupby("Phase")["Participant"].nunique().reindex(PHASE_ORDER)
    return summary, counts


def build_table(kit):
    summary, counts = summarise(kit)
    summary.to_csv(os.path.join(OUTPUT_DIR, f"table4_{kit}_summary.csv"))
    sigs = list(SIGNALS[kit])
    header = "Phase" + (" & $n$" if SHOW_N else "") + " & Duration" + "".join(f" & {SIGNALS[kit][s]['label']} ({SIGNALS[kit][s]['unit']})" for s in sigs) + r" \\"
    body = []
    for phase_key, info in PHASES.items():
        cells = [info["label"]]
        if SHOW_N:
            cells.append(str(int(counts[phase_key])) if pd.notna(counts[phase_key]) else "0")
        cells.append(fmt_duration(summary.loc[phase_key, ("Duration_s", "mean")], summary.loc[phase_key, ("Duration_s", "std")]))
        cells += [fmt_value(summary.loc[phase_key, (s, "mean")], summary.loc[phase_key, (s, "std")]) for s in sigs]
        body.append(" & ".join(cells) + r" \\")
    align = "l" + "c" * (1 + len(sigs) + (1 if SHOW_N else 0))
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{CAPTIONS[kit]}}}", rf"\label{{{LABELS[kit]}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule", header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


gsr_tex, ppg_tex = build_table("GSR"), build_table("PPG")          # GSR first, then PPG (as in the reference image)
for name, tex in (("table_GSR.tex", gsr_tex), ("table_PPG.tex", ppg_tex), ("tables_raw_summary.tex", gsr_tex + "\n" + ppg_tex)):
    with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
        f.write(tex)

# ============================================================
# 11. FINAL REPORT
# ============================================================
print("\n========== TABLES GENERATED ==========")
print("Requested participants:", len(PARTICIPANTS), "| unique:", len(set(PARTICIPANTS)))
if not P.empty:
    print("\nParticipants per kit and phase:")
    print(P.groupby(["Kit", "Phase"])["Participant"].nunique().unstack().reindex(columns=PHASE_ORDER))
print("\nMissing recordings:", len(missing))
print("\n" + gsr_tex + "\n" + ppg_tex)
print("Output folder:", OUTPUT_DIR)
