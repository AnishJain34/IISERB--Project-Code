"""
GSR / EDA feature extraction - rewritten pipeline for the phase-file folder structure

    INPUT_FOLDER\\P7\\P7_Baseline_EDA.csv
                      P7_2BackPart1_EDA.csv
                      P7_2BackPart2_EDA.csv
                      P7_SpeechRest_Stressed_EDA.csv   (or ..._Control_EDA.csv)
                      

Only the GSR skin-conductance column is used (P7_GSR_Skin_Conductance_CAL). Other columns
(PPG, temperature, pressure, GSR range / resistance) are ignored.

Same feature set and same feature names as the original pipeline, plus these fixes
  * `src.find_sampling_rate` was undefined -> sampling rate is taken from the timestamps.
  * Timestamps like "18-11-2024 15:02:49.002" are parsed day-first.
  * The 512 Hz stream is put on a regular 16 Hz TIME grid (mean per 1/16 s bin) so timestamp gaps do not
    shift the window times; gaps up to MAX_INTERPOLATION_GAP_SEC are interpolated, longer ones leave NaN
    and the window is skipped (status = GAP_IN_WINDOW) instead of being filtered across the gap.
  * `scipy.integrate.trapz` (removed in new SciPy) -> `trapezoid`.
  * Windows with no SCR peaks give NaN statistics instead of crashing.
  * No os.chdir(); phase/summary files that are not EDA recordings are skipped.
Output: OUTPUT_FOLDER\\P7\\P7_Baseline_EDA_features.csv  (one row per window)
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.signal import signal_sanitize
from scipy.integrate import trapezoid

try:
    import pywt
except ImportError:                      # wavelet features are skipped if PyWavelets is missing
    pywt = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================ CONFIG ============================
INPUT_FOLDER = Path(os.environ.get("EDA_IN", r"D:\automate\EDA_Phase"))
OUTPUT_FOLDER = Path(os.environ.get("EDA_OUT", r"D:\EDA_DATA\Features_EDA_CVXEDA"))
TARGET_SAMPLING_RATE = 16
MODE = "window"                          # "window" or "full"
WINDOW_SIZE_SEC = 60
OVERLAP_SEC = 45
MAX_INTERPOLATION_GAP_SEC = 2.0
NUMERIC_TIMESTAMP_UNIT = "ms"            # only used if the timestamp column is numeric


# ============================ GSR FEATURE FUNCTIONS ============================
def clean_and_decompose_eda(sig, sr):
    eda_signal = signal_sanitize(sig)
    return nk.eda_clean(eda_signal, sampling_rate=sr, method="biosppy")


def _stats(prefix, values, features):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    ok = len(v) > 0
    features[prefix + "_mean"] = float(np.mean(v)) if ok else np.nan
    features[prefix + "_max"] = float(np.max(v)) if ok else np.nan
    features[prefix + "_min"] = float(np.min(v)) if ok else np.nan
    features[prefix + "_std"] = float(np.std(v)) if ok else np.nan


def feature_extraction(eda_cleaned, df_eda_mav, sr):
    sampling_rate = sr
    eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate,method="cvxeda")
    phasic = eda_decomposed["EDA_Phasic"].values
    tonic = eda_decomposed["EDA_Tonic"].values

    _, info = nk.eda_peaks(phasic, sampling_rate=sampling_rate, method="nabian2018")
    info1 = nk.eda_findpeaks(phasic, sampling_rate=sampling_rate, method="nabian2018")

    def arr(d, key):
        return np.asarray(d.get(key, []), dtype=float).reshape(-1)

    onsets, peaks = arr(info, "SCR_Onsets"), arr(info, "SCR_Peaks")
    heights = arr(info, "SCR_Height")
    offsets = arr(info1, "SCR_Offset") if "SCR_Offset" in info1 else arr(info, "SCR_Recovery")
    m = min(len(onsets), len(peaks), len(offsets))
    onsets_a, peaks_a, offsets_a = onsets[:m], peaks[:m], offsets[:m]
    ok = np.isfinite(onsets_a) & np.isfinite(peaks_a) & np.isfinite(offsets_a)

    features = {}
    _stats("SCR_Duration", (offsets_a[ok] - onsets_a[ok]) / sr, features)
    _stats("SCR_Amplitude", heights, features)
    _stats("SCR_RiseTime", (peaks_a[ok] - onsets_a[ok]) / sr, features)
    _stats("SCR_RecoveryTime", (offsets_a[ok] - peaks_a[ok]) / sr, features)
    features["SCR_Peak_count"] = int(np.isfinite(peaks).sum())

    _stats("SCR_raw", df_eda_mav, features)
    _stats("SC", eda_cleaned, features)
    _stats("SCL", tonic, features)

    features["phasic_median"] = np.median(phasic)
    features["phasic_dynamic_range"] = np.max(phasic) - np.min(phasic)
    features["phasic_slope"] = np.polyfit(np.arange(len(phasic)), phasic, 1)[0]

    first_derivative = np.diff(phasic)
    second_derivative = np.diff(first_derivative)
    features["phasic_first_derivative_mean"] = np.mean(first_derivative)
    features["phasic_first_derivative_std"] = np.std(first_derivative)
    features["phasic_second_derivative_mean"] = np.mean(second_derivative)
    features["phasic_second_derivative_std"] = np.std(second_derivative)

    scr_areas = []
    for onset, offset in zip(onsets_a[ok].astype(int), offsets_a[ok].astype(int)):
        if offset > onset:
            scr_areas.append(trapezoid(phasic[onset:offset], dx=1 / sampling_rate))
    features["SCR_mean_area"] = float(np.mean(scr_areas)) if scr_areas else 0

    if pywt is not None:
        for freq in [4, 2, 1]:
            coeffs = pywt.wavedec(phasic, "db4", level=int(np.log2(sampling_rate / freq)))
            features[f"phasic_wavelet_{freq}Hz_mean"] = np.mean(coeffs[-1])
            features[f"phasic_wavelet_{freq}Hz_median"] = np.median(coeffs[-1])
            features[f"phasic_wavelet_{freq}Hz_std"] = np.std(coeffs[-1])

    return pd.DataFrame([features])


# ============================ COLUMN HELPERS ============================
def find_timestamp_column(columns):
    for col in columns:
        if "timestamp" in str(col).lower():
            return col
    return None


def find_gsr_column(columns):
    for col in columns:
        if "gsr_skin_conductance" in str(col).lower():
            return col
    return None


# ============================ LOADER (16 Hz time grid) ============================
def load_and_downsample_gsr(file_path, target_sampling_rate=TARGET_SAMPLING_RATE):
    """Read ONE recording, keep timestamp + GSR_Skin_Conductance, return the 16 Hz signal."""
    header = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig").columns
    time_col, gsr_col = find_timestamp_column(header), find_gsr_column(header)
    if time_col is None or gsr_col is None:
        raise KeyError("no Timestamp / GSR_Skin_Conductance column (not an EDA recording)")

    df = pd.read_csv(file_path, usecols=[time_col, gsr_col], encoding="utf-8-sig")
    if pd.api.types.is_numeric_dtype(df[time_col]):
        t = pd.to_datetime(df[time_col], unit=NUMERIC_TIMESTAMP_UNIT, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(df[time_col], dayfirst=True, utc=True, errors="coerce")
    df = pd.DataFrame({"gsr": pd.to_numeric(df[gsr_col], errors="coerce").to_numpy()}, index=t)
    df = df[df.index.notna()].dropna().sort_index()
    if df.empty:
        raise ValueError("no valid timestamp/GSR rows")
    df = df.groupby(level=0).mean()                                  # duplicate timestamps

    original_sr = 1.0 / float(np.median(np.diff(df.index.view("int64") / 1e9)))   # info only
    step = pd.to_timedelta(1 / target_sampling_rate, unit="s")
    regular = df["gsr"].resample(step, origin="start").mean()
    limit = max(1, int(round(MAX_INTERPOLATION_GAP_SEC * target_sampling_rate)))
    regular = regular.interpolate(method="time", limit=limit, limit_area="inside")
    gsr_signal = regular.to_numpy(dtype=float)
    info = {"original_sr_est": original_sr, "n_missing_after_interp": int(np.isnan(gsr_signal).sum())}
    return gsr_signal, target_sampling_rate, info


# ============================ EXTRACTION MODES ============================
def process_full_segment(gsr_signal, sampling_rate):
    if np.isnan(gsr_signal).any():
        raise ValueError("signal contains gaps longer than MAX_INTERPOLATION_GAP_SEC; use mode='window'")
    eda_cleaned = clean_and_decompose_eda(gsr_signal, sampling_rate)
    feats = feature_extraction(eda_cleaned, gsr_signal, sampling_rate)
    feats.insert(0, "window_index", 0)
    feats.insert(1, "window_start_sec", 0.0)
    feats.insert(2, "window_end_sec", round(len(gsr_signal) / sampling_rate, 3))
    feats["processing_status"] = "OK"
    return feats


def process_windowed(gsr_signal, sampling_rate, window_size_sec=WINDOW_SIZE_SEC, overlap_sec=OVERLAP_SEC):
    if overlap_sec >= window_size_sec:
        raise ValueError("overlap_sec must be strictly less than window_size_sec.")
    window_samples = int(window_size_sec * sampling_rate)
    step_samples = int((window_size_sec - overlap_sec) * sampling_rate)
    rows, win_idx, start = [], 0, 0
    while start + window_samples <= len(gsr_signal):
        end = start + window_samples
        segment = gsr_signal[start:end]
        row = {"window_index": win_idx, "window_start_sec": round(start / sampling_rate, 3),
               "window_end_sec": round(end / sampling_rate, 3)}
        if np.isnan(segment).any():
            row["processing_status"] = "GAP_IN_WINDOW"
        else:
            try:
                eda_cleaned = clean_and_decompose_eda(segment, sampling_rate)
                row.update(feature_extraction(eda_cleaned, segment, sampling_rate).iloc[0].to_dict())
                row["processing_status"] = "OK"
            except Exception as e:
                row["processing_status"] = "FAILED:" + str(e)[:100]
                print(f"    Window {win_idx} ({row['window_start_sec']}s-{row['window_end_sec']}s) failed: {e}")
        rows.append(row)
        start += step_samples
        win_idx += 1
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)                                          # failed windows -> NaN features
    lead = ["window_index", "window_start_sec", "window_end_sec"]
    feats = [c for c in df.columns if c not in lead + ["processing_status"]]
    return df[lead + feats + ["processing_status"]]


# ============================ BATCH ============================
def extract_gsr_features_for_all_participants(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER,
                                              target_sampling_rate=TARGET_SAMPLING_RATE, mode=MODE,
                                              window_size_sec=WINDOW_SIZE_SEC, overlap_sec=OVERLAP_SEC):
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    pid_folders = sorted(d for d in input_folder.iterdir() if d.is_dir())
    if not pid_folders:
        print("No participant subfolders found in the input folder.")
        return
    summary = []
    for pid_dir in pid_folders:
        print(f"\n{'=' * 60}\n  Participant: {pid_dir.name}\n{'=' * 60}")
        csv_files = sorted(p for p in pid_dir.rglob("*") if p.suffix.lower() == ".csv"
                           and not p.name.lower().endswith("_features.csv"))
        for file_path in csv_files:
            rel = file_path.relative_to(pid_dir)
            print(f"\n  Processing : {rel}")
            try:
                gsr_signal, sr, info = load_and_downsample_gsr(file_path, target_sampling_rate)
                if mode == "full":
                    result = process_full_segment(gsr_signal, sr)
                elif mode == "window":
                    result = process_windowed(gsr_signal, sr, window_size_sec, overlap_sec)
                else:
                    raise ValueError(f"Unknown mode '{mode}'. Choose 'full' or 'window'.")
                if result.empty:
                    print("  Warning: no complete window (recording shorter than window size).")
                    continue
                result.insert(0, "source_file", str(rel))
                out_dir = output_folder / pid_dir.name / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / (file_path.stem + "_features.csv")
                result.to_csv(out_path, index=False)
                st = result["processing_status"].str.split(":").str[0].value_counts().to_dict()
                print(f"  Saved      -> {out_path}   windows={len(result)} {st}")
                summary.append({"participant": pid_dir.name, "file": str(rel), "windows": len(result),
                                "samples_missing_after_interp": info["n_missing_after_interp"], **st})
            except KeyError as e:
                print(f"  SKIPPED    : {e}")
            except Exception as e:
                print(f"  FAILED     : {e}")
                summary.append({"participant": pid_dir.name, "file": str(rel), "error": str(e)[:200]})
    output_folder.mkdir(parents=True, exist_ok=True)
    if summary:
        pd.DataFrame(summary).to_csv(output_folder / "00_EDA_EXTRACTION_SUMMARY.csv", index=False)
    print(f"\nDone. All feature files saved under: {output_folder}")


if __name__ == "__main__":
    extract_gsr_features_for_all_participants()