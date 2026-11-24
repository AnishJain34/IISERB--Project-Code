# ==============================================================================
# PPG HRV FEATURE EXTRACTION (EXACT SAME FORMAT AS NILESH'S PIPELINE)
#
# Output CSV Format:
#   source_file | window_index | window_start_sec | window_end_sec | HRV_MeanNN ...
#
# Output File Structure:
#   output_folder / PID / subfolder / <original_name>_features.csv
# ==============================================================================

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk

warnings.filterwarnings("ignore")


# ---------------- Feature Functions (Pure NeuroKit2) ---------------- #

def hrv_feature_time(rpeaks, sr):
    return nk.hrv_time(rpeaks, sampling_rate=sr)


def hrv_feature_freq(rpeaks, sr):
    return nk.hrv_frequency(rpeaks, sampling_rate=sr, psd_method="welch")


def hrv_feature_nonlinear(rpeaks, sr):
    return nk.hrv_nonlinear(rpeaks, sampling_rate=sr)


def hrv_feature_rqa(rpeaks, sr):
    try:
        return nk.hrv_rqa(rpeaks, sampling_rate=sr)
    except Exception:
        return pd.DataFrame()


def determine_ppg_peaks(ppg_signal, sr):
    """
    Standard PPG peak detection using Elgendi method.
    Replaces ecg.hamilton_segmenter to properly detect systolic peaks on PPG.
    """
    # 1. Bandpass filter & baseline removal optimized for PPG (0.5 - 8.0 Hz)
    clean_signal = nk.ppg_clean(ppg_signal, sampling_rate=sr, method="elgendi")

    # 2. Systolic peak detection
    _, peak_info = nk.ppg_peaks(clean_signal, sampling_rate=sr, method="elgendi", show=False)
    peaks = np.unique(np.asarray(peak_info.get("PPG_Peaks", []), dtype=int))
    peaks = peaks[(peaks >= 0) & (peaks < len(clean_signal))]

    return clean_signal, peaks


# ---------------- Column Auto-Detection Helpers ---------------- #

def find_timestamp_column(df):
    """Return the column whose name contains 'Timestamp' (case-insensitive)."""
    for col in df.columns:
        if "timestamp" in col.lower():
            return col
    for col in df.columns:
        if col.lower() in {"time", "datetime", "utctime"}:
            return col
    raise ValueError(f"No Timestamp column found. Available columns: {df.columns.tolist()[:10]}")


def find_ppg_column(df, time_col):
    """
    Auto-detect raw PPG signal column.
    Prefers PPG_A13 (e.g. P7_PPG_A13_CAL) and avoids device-calculated heart rate columns.
    """
    # First look for PPG_A13
    for col in df.columns:
        if col != time_col and "ppg" in col.lower() and "a13" in col.lower():
            if pd.to_numeric(df[col], errors="coerce").notna().sum() > 10:
                return col

    # Fallback to any PPG column that is not device HR
    for col in df.columns:
        if col != time_col and "ppg" in col.lower() and not any(x in col.lower() for x in ["hr", "ibi"]):
            if pd.to_numeric(df[col], errors="coerce").notna().sum() > 10:
                return col

    raise ValueError(f"No valid raw PPG column found. Available columns: {df.columns.tolist()[:10]}")


# ---------------- Signal Loader & Sampling Rate ---------------- #

def load_ppg_signal(file_path):
    """
    Load a single PPG CSV and auto-detect Timestamp, PPG column, and Sampling Rate.
    Replaces the local 'src' (sampling_rate_calculator) dependency.
    """
    df = pd.read_csv(file_path, low_memory=False)

    time_col = find_timestamp_column(df)
    ppg_col = find_ppg_column(df, time_col)

    # 1. Parse timestamps reliably
    numeric = pd.to_numeric(df[time_col], errors="coerce")
    if numeric.notna().mean() > 0.95:
        med = float(numeric.dropna().abs().median())
        unit = "ms" if med > 1e11 else "s"
        timestamps = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    else:
        text = df[time_col].astype(str).str.strip()
        try:
            timestamps = pd.to_datetime(text, format="mixed", dayfirst=True, errors="coerce", utc=True)
        except Exception:
            timestamps = pd.to_datetime(text, dayfirst=True, errors="coerce", utc=True)

    df["_parsed_time"] = timestamps
    df = df.dropna(subset=["_parsed_time", ppg_col]).sort_values("_parsed_time")

    # 2. Compute sampling rate
    seconds = (df["_parsed_time"] - df["_parsed_time"].iloc[0]).dt.total_seconds().to_numpy()
    diffs = np.diff(seconds)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Invalid timestamps: cannot estimate sampling rate.")

    sampling_rate = float(round(1.0 / np.median(diffs), 2))

    # 3. Regularize signal grid (interpolating minor sensor jitter)
    count = int(np.floor(seconds[-1] * sampling_rate)) + 1
    grid = np.arange(count) / sampling_rate
    ppg_signal = np.interp(grid, seconds, pd.to_numeric(df[ppg_col], errors="coerce").to_numpy())

    return ppg_signal, sampling_rate


# ---------------- HRV Extraction Modes ---------------- #

def extract_hrv_from_signal(ppg_signal, sampling_rate):
    """Extract all HRV features from a PPG segment. Returns a one-row DataFrame."""
    _, rpeaks = determine_ppg_peaks(ppg_signal, sampling_rate)

    if len(rpeaks) < 5:
        raise ValueError(f"Too few pulse peaks detected ({len(rpeaks)} peaks).")

    # Filter out non-physiological inter-beat intervals (<30 BPM or >200 BPM)
    ibi = np.diff(rpeaks) / sampling_rate
    valid_mask = (ibi >= 0.30) & (ibi <= 2.00)
    if valid_mask.sum() < 5:
        raise ValueError("Too few valid inter-beat intervals in physiological range.")

    t_df = hrv_feature_time(rpeaks, sampling_rate)
    f_df = hrv_feature_freq(rpeaks, sampling_rate)
    nl_df = hrv_feature_nonlinear(rpeaks, sampling_rate)
    rqa_df = hrv_feature_rqa(rpeaks, sampling_rate)

    feats = pd.concat([t_df, f_df, nl_df, rqa_df], axis=1)
    return feats


def process_full_segment(ppg_signal, sampling_rate):
    """
    Extract HRV features from the entire PPG signal as one segment.
    Returns a one-row DataFrame.
    """
    feats = extract_hrv_from_signal(ppg_signal, sampling_rate)
    feats.insert(0, "window_index", 0)
    feats.insert(1, "window_start_sec", 0.0)
    feats.insert(2, "window_end_sec", round(len(ppg_signal) / sampling_rate, 3))
    return feats


def process_windowed(ppg_signal, sampling_rate, window_size_sec=60, overlap_sec=45):
    """
    Extract HRV features using a sliding window approach.
    Returns a DataFrame with one row per window matching the Nilesh pipeline format.
    """
    if overlap_sec >= window_size_sec:
        raise ValueError("overlap_sec must be strictly less than window_size_sec.")

    window_samples = int(window_size_sec * sampling_rate)
    step_samples = int((window_size_sec - overlap_sec) * sampling_rate)

    all_rows = []
    feature_columns = None
    win_idx = 0
    start = 0

    while start + window_samples <= len(ppg_signal):
        end = start + window_samples
        segment = ppg_signal[start:end]
        start_sec = round(start / sampling_rate, 3)
        end_sec = round(end / sampling_rate, 3)

        try:
            feats = extract_hrv_from_signal(segment, sampling_rate)

            if feature_columns is None:
                feature_columns = feats.columns.tolist()

            row = feats.iloc[0].to_dict()
            row["window_index"] = win_idx
            row["window_start_sec"] = start_sec
            row["window_end_sec"] = end_sec
            all_rows.append(row)

        except Exception as e:
            # When peak detection fails in a noisy window, record NaN to maintain window indexing
            if feature_columns is not None:
                nan_row = {col: np.nan for col in feature_columns}
                nan_row["window_index"] = win_idx
                nan_row["window_start_sec"] = start_sec
                nan_row["window_end_sec"] = end_sec
                all_rows.append(nan_row)

        start += step_samples
        win_idx += 1

    if not all_rows:
        return pd.DataFrame()

    final_df = pd.DataFrame(all_rows)
    cols_order = ["window_index", "window_start_sec", "window_end_sec"] + [
        c for c in feature_columns if c in final_df.columns
    ]
    final_df = final_df[cols_order]
    return final_df


# ---------------- Main Batch Pipeline ---------------- #

def extract_ppg_hrv_features_for_all_participants(
    input_folder,
    output_folder,
    mode="window",
    window_size_sec=60,
    overlap_sec=45
):
    """
    Walks through input_folder/PID, extracts HRV features from raw PPG,
    and saves each CSV as '<original_name>_features.csv' under output_folder/PID.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    pid_folders = sorted([
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not pid_folders:
        print("No participant subfolders found in the input folder.")
        return

    for pid_dir in pid_folders:
        pid = pid_dir.name
        pid_output_path = output_path / pid
        pid_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Participant: {pid}")
        print(f"{'='*60}")

        csv_files = sorted(pid_dir.rglob("*.csv"))
        if not csv_files:
            print(f"  No CSV files found under {pid_dir}")
            continue

        for file_path in csv_files:
            if file_path.name.endswith("_features.csv"):
                continue

            rel_path = file_path.relative_to(pid_dir)
            out_name = f"{file_path.stem}_features.csv"
            out_dir = pid_output_path / rel_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / out_name

            print(f"  Processing : {rel_path}")

            try:
                ppg_signal, sampling_rate = load_ppg_signal(file_path)

                if mode == "full":
                    result_df = process_full_segment(ppg_signal, sampling_rate)
                elif mode == "window":
                    result_df = process_windowed(
                        ppg_signal,
                        sampling_rate,
                        window_size_sec=window_size_sec,
                        overlap_sec=overlap_sec
                    )
                else:
                    raise ValueError(f"Unknown mode '{mode}'. Choose 'full' or 'window'.")

                if result_df.empty:
                    print("    Warning: No windows extracted.")
                    continue

                # Insert source_file at index 0 to match Nilesh's format exactly
                result_df.insert(0, "source_file", str(rel_path).replace("\\", "/"))
                result_df.to_csv(out_file, index=False)
                print(f"    Saved ({len(result_df)} windows) -> {out_file.name}")

            except Exception as e:
                print(f"    FAILED : {e}")

    print(f"\nDone. All feature files saved under: {output_folder}")


# ---------------- Execution ---------------- #

if __name__ == "__main__":
    extract_ppg_hrv_features_for_all_participants(
        input_folder    = r"D:\automate\EDA_Phase",
        output_folder   = r"D:\EDA_DATA\Features_PPG",
        mode            = "window",
        window_size_sec = 60,    # 60-second window
        overlap_sec     = 45     # 45-second overlap (15-second step)
    )
