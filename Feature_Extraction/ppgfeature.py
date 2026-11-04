from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import neurokit2 as nk

warnings.filterwarnings("ignore")

INPUT_FOLDER = Path(r"D:\automate\EDA_Phase")
OUTPUT_FOLDER = Path(r"D:\EDA_DATA\Features_PPG")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 60
OVERLAP_SEC = 45
STEP_SEC = WINDOW_SEC - OVERLAP_SEC
FIXED_FS = None                 # Set to a number if sampling rate is known
MAX_MISSING_FRACTION = 0.50
MIN_VALID_IBI = 5
MIN_IBI_SEC, MAX_IBI_SEC = 0.30, 2.00
MIN_FS_HZ = 20.0                # a PPG sampling rate below this means the timestamps are wrong

# Raw PPG signal = the "PPG_A13" column (e.g. P7_PPG_A13_CAL).  If it is missing the file
# FAILS instead of silently using PPG_IBI / PPGtoHR (which are NOT raw signals).
PPG_SIGNAL_KEY = "ppga13"       # normalised name fragment of the raw PPG column

SKIP_FILES = {
    "ppg_all_participants_all_phases.csv",
    "ppg_file_audit.csv",
    "ppg_failed_files.csv",
}


# ------------------------- SMALL HELPERS -------------------------

def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, names):
    """Exact (normalised) name match."""
    lookup = {norm(c): c for c in df.columns}
    return next((lookup[norm(n)] for n in names if norm(n) in lookup), None)


def find_timestamp_column(df):
    """
    Works with participant-prefixed Shimmer names such as
    'P7_Timestamp_FormattedUnix_CAL'.  A column whose name contains 'timestamp' is used;
    the human-readable 'Formatted' column is preferred over a raw Unix number.
    """
    candidates = [c for c in df.columns if "timestamp" in norm(c)]
    if not candidates:
        candidates = [c for c in df.columns
                      if norm(c) in {"time", "datetime", "utctime"}]
    if not candidates:
        return None
    formatted = [c for c in candidates if "formatted" in norm(c)]
    return (formatted or candidates)[0]


def find_ppg_column(df, timestamp_col):
    """Raw PPG signal: the column containing 'PPG_A13' (e.g. P7_PPG_A13_CAL)."""
    matches = [c for c in df.columns if c != timestamp_col and PPG_SIGNAL_KEY in norm(c)]
    for col in matches:
        if pd.to_numeric(df[col], errors="coerce").notna().sum() > 10:
            return col
    return None


def find_optional_column(df, fragment, exclude=()):
    for col in df.columns:
        n = norm(col)
        if fragment in n and not any(e in n for e in exclude):
            return col
    return None


def column_participant_prefix(columns):
    """'P7_PPG_A13_CAL' -> 'P7' (used only to cross-check the folder name)."""
    for c in columns:
        m = re.match(r"^(P\d+)_", str(c), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def parse_time(series):
    """Formatted strings (day-first, as in the Arya exports) or numeric Unix time."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:                         # numeric Unix timestamp
        med = float(numeric.dropna().abs().median())
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    text = series.astype("string").str.strip()
    try:
        parsed = pd.to_datetime(text, format="mixed", dayfirst=True, errors="coerce", utc=True)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce", utc=True)
    return parsed


def participant_phase(path):
    rel = path.relative_to(INPUT_FOLDER)
    participant = rel.parts[0] if len(rel.parts) > 1 else "Unknown"
    name = path.stem.lower()

    if "baseline" in name:
        phase = "Baseline"
    elif "speech" in name and ("rest" in name or "stress" in name):
        phase = "SpeechRest"
    elif any(x in name for x in ("2backpart1", "2back_part1", "2back1")):
        phase = "2Backpart1"
    elif any(x in name for x in ("2backpart2", "2back_part2", "2back2")):
        phase = "2Backpart2"
    elif "2back" in name:
        phase = "2Back"
    elif "summary" in name:
        phase = "Summary"
    else:
        phase = path.stem
    return participant, phase


def load_recording(path):
    df = pd.read_csv(path, low_memory=False)

    time_col = find_timestamp_column(df)
    if time_col is None:
        raise ValueError(f"Timestamp column not found. Columns: {list(df.columns)[:12]}")

    ppg_col = find_ppg_column(df, time_col)
    if ppg_col is None:
        raise ValueError("Raw PPG column (PPG_A13) not found or empty. "
                         f"Columns: {list(df.columns)[:12]}")

    # optional device-computed heart rate (Shimmer PPGtoHR) for quality control only
    hr_col = find_optional_column(df, "ppgtohr")

    frame = pd.DataFrame({
        "time": parse_time(df[time_col]),
        "ppg": pd.to_numeric(df[ppg_col], errors="coerce"),
    })
    if hr_col is not None:
        hr = pd.to_numeric(df[hr_col], errors="coerce")
        frame["device_hr"] = hr.where((hr >= 30) & (hr <= 220))   # drop invalid / sentinel values

    n_bad_time = int(frame["time"].isna().sum())
    frame = frame.dropna(subset=["time", "ppg"]).sort_values("time")
    agg = {"ppg": "mean"}
    if "device_hr" in frame:
        agg["device_hr"] = "mean"
    data = frame.groupby("time", as_index=False).agg(agg)

    if len(data) < 100:
        raise ValueError(f"Too few valid PPG samples (unparsed timestamps: {n_bad_time})")

    t = data.time
    # total_seconds() is independent of the datetime resolution (pandas >= 3 no longer defaults to ns)
    seconds = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    diffs = np.diff(seconds)
    diffs = diffs[diffs > 0]
    if not len(diffs):
        raise ValueError("Invalid timestamps")

    fs = float(FIXED_FS or 1 / np.median(diffs))
    if fs < MIN_FS_HZ:
        raise ValueError(f"Implausible PPG sampling rate {fs:.2f} Hz - check timestamp column "
                         "(whole-second timestamps cannot be used)")
    duration = seconds[-1]
    count = int(np.floor(duration * fs)) + 1
    grid = np.arange(count) / fs
    signal = np.interp(grid, seconds, data.ppg.to_numpy())

    typical = np.median(diffs)
    gaps = np.diff(seconds)
    missing_fraction = min(1.0, gaps[gaps > max(2.5 / fs, 2.5 * typical)].sum() / duration)

    qc = None
    if "device_hr" in data and data["device_hr"].notna().sum() > 10:
        qc = (seconds, data["device_hr"].to_numpy())
    return signal, fs, t.iloc[0], missing_fraction, ppg_col, qc


# ------------------------- DYNAMIC FEATURE EXTRACTION -------------------------

def add_neurokit_features(features, peaks, fs):
    """Add all columns returned by the installed NeuroKit2 version."""
    peak_info = {"PPG_Peaks": np.asarray(peaks, dtype=int)}
    for function_name in ("hrv_time", "hrv_frequency", "hrv_nonlinear", "hrv_rqa"):
        function = getattr(nk, function_name, None)
        if function is None:
            continue
        try:
            result = function(peak_info, sampling_rate=fs, show=False)
            if isinstance(result, pd.DataFrame) and not result.empty:
                features.update(result.iloc[0].to_dict())
        except Exception:
            pass


def extract_features(window, fs):
    x = np.asarray(window, dtype=float)
    features = {
        "PPG_Raw_Sample_Count": len(x),
        "Missing_Data_Percent": float((~np.isfinite(x)).mean() * 100)
    }

    if len(x) < int(WINDOW_SEC * fs * 0.8):
        return features, "FAILED", "Too few samples"

    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) < 10 or features["Missing_Data_Percent"] / 100 > MAX_MISSING_FRACTION:
        return features, "FAILED", "Insufficient valid signal"

    x = np.interp(np.arange(len(x)), valid, x[valid])
    try:
        clean = nk.ppg_clean(x, sampling_rate=fs, method="elgendi")
        _, info = nk.ppg_peaks(clean, sampling_rate=fs, method="elgendi", show=False)
        peaks = np.unique(np.asarray(info.get("PPG_Peaks", []), dtype=int))
        peaks = peaks[(peaks >= 0) & (peaks < len(clean))]
    except Exception as exc:
        return features, "FAILED", f"PPG processing failed: {exc}"

    features.update({
        "PPG_Clean_Sample_Count": len(clean),
        "PPG_Clean_Mean": float(np.mean(clean)),
        "PPG_Clean_SD": float(np.std(clean, ddof=1)),
        "PPG_Pulse_Peak_Count": len(peaks),
        "PPG_Peak_Rate_per_sec": len(peaks) / (len(clean) / fs)
    })

    # Pulse-to-pulse intervals and rate statistics
    ibi = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
    valid_ibi = ibi[(ibi >= MIN_IBI_SEC) & (ibi <= MAX_IBI_SEC)]
    features["IBI_Raw_Count"] = len(ibi)
    features["IBI_Valid_Count"] = len(valid_ibi)

    if len(valid_ibi):
        hr = 60 / valid_ibi
        features.update({
            "PPI_ms_mean": float(np.mean(valid_ibi) * 1000),
            "HR_mean": float(np.mean(hr)),
            "PPG_Heart_Rate_Min_bpm": float(np.min(hr)),
            "PPG_Heart_Rate_Max_bpm": float(np.max(hr))
        })
        if len(valid_ibi) > 1:
            features.update({
                "PPI_ms_std": float(np.std(valid_ibi, ddof=1) * 1000),
                "HR_std": float(np.std(hr, ddof=1)),
                "HR_Acceleration": float(np.mean(np.diff(hr))),
                "HRV_RMSSD_ms": float(np.sqrt(np.mean(np.diff(valid_ibi * 1000) ** 2)))
            })

    # NeuroKit2 expects consecutive peaks. Use it only when every interval is valid,
    # avoiding artificial intervals across rejected gaps.
    if len(valid_ibi) >= MIN_VALID_IBI and len(valid_ibi) == len(ibi):
        add_neurokit_features(features, peaks, fs)

    # Pulse morphology: trough-to-peak rise and peak-to-next-peak fall
    morphology = {"amplitude": [], "rise": [], "fall": [], "duration": [], "downstroke": []}
    for i in range(1, len(peaks) - 1):
        prev_peak, peak, next_peak = peaks[i - 1:i + 2]
        trough = prev_peak + np.argmin(clean[prev_peak:peak + 1])
        if trough >= peak:
            continue
        rise, fall = (peak - trough) / fs, (next_peak - peak) / fs
        morphology["amplitude"].append(clean[peak] - clean[trough])
        morphology["rise"].append(rise)
        morphology["fall"].append(fall)
        morphology["duration"].append(rise + fall)
        if fall > 0:
            morphology["downstroke"].append((clean[next_peak] - clean[peak]) / fall)

    for name, values in morphology.items():
        if values:
            label = {
                "amplitude": "PPG_Amplitude",
                "rise": "PPG_RiseTime_s",
                "fall": "PPG_FallTime_s",
                "duration": "PPG_PulseDuration_s",
                "downstroke": "PPG_DownstrokeSlope"
            }[name]
            features[f"{label}_mean"] = float(np.mean(values))
            if len(values) > 1:
                features[f"{label}_std"] = float(np.std(values, ddof=1))

    first = np.diff(clean) * fs
    second = np.diff(first) * fs
    features["PPG_FirstDerivative_std"] = float(np.std(first, ddof=1))
    features["PPG_SecondDerivative_std"] = float(np.std(second, ddof=1))

    if len(peaks) < 2:
        return features, "WARNING", "Too few detected pulse peaks"
    return features, "OK", ""


# ------------------------- PROCESS ONE FILE -------------------------

def process_file(path):
    participant, phase = participant_phase(path)
    signal, fs, start_time, missing, ppg_col, qc = load_recording(path)
    prefix = column_participant_prefix([ppg_col])
    if prefix and participant != "Unknown" and prefix.upper() != participant.upper():
        print(f"  WARNING: folder says {participant} but column prefix says {prefix}")
    window_n, step_n = round(WINDOW_SEC * fs), round(STEP_SEC * fs)
    rows = []

    for start in range(0, len(signal), step_n):
        end = min(start + window_n, len(signal))
        if end - start < window_n * 0.8:
            break

        features, status, message = extract_features(signal[start:end], fs)
        start_sec, end_sec = start / fs, end / fs

        # QC: compare our PPG-derived heart rate with the device's own PPGtoHR value
        device_hr = np.nan
        if qc is not None:
            t_sec, hr_dev = qc
            lo, hi = np.searchsorted(t_sec, [start_sec, end_sec])
            if hi - lo > 10 and np.isfinite(hr_dev[lo:hi]).any():
                device_hr = float(np.nanmean(hr_dev[lo:hi]))
        hr_diff = (features["HR_mean"] - device_hr
                   if "HR_mean" in features and np.isfinite(device_hr) else np.nan)

        row = {
            "Participant": participant,
            "Phase": phase,
            "Source_File": path.name,
            "PPG_Source_Column": ppg_col,
            "Window_Index": len(rows),
            "window_start_sec": start_sec,
            "window_end_sec": end_sec,
            "Window_Start": (start_time + pd.to_timedelta(start_sec, unit="s")).isoformat(),
            "Window_End": (start_time + pd.to_timedelta(end_sec, unit="s")).isoformat(),
            "Sampling_Rate_Hz": fs,
            "Window_Duration_s": (end - start) / fs,
            "Recording_Missing_Data_Percent": missing * 100,
            "Device_HR_mean_bpm": device_hr,
            "HR_minus_Device_HR_bpm": hr_diff,
            "Processing_Status": status,
            "Processing_Message": message,
            **features
        }
        rows.append(row)
        if end == len(signal):
            break

    return pd.DataFrame(rows), participant, phase, ppg_col, fs


# ------------------------- TEST ONE PARTICIPANT -------------------------

# ------------------------- TEST ALL PARTICIPANTS -------------------------

def main():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(INPUT_FOLDER)

    files = [
        p for p in sorted(INPUT_FOLDER.rglob("*.csv"))
        if p.name.lower() not in SKIP_FILES
        and not p.name.lower().endswith("_features.csv")
        and "phase_summary" not in p.name.lower()
        and OUTPUT_FOLDER.resolve() not in p.resolve().parents
    ]

    all_rows = []
    audit = []
    failed = []

    print(f"Total raw files found: {len(files)}")

    for i, path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {path}")

        try:
            result, participant, phase, ppg_col, fs = process_file(path)

            folder = OUTPUT_FOLDER / participant
            folder.mkdir(parents=True, exist_ok=True)

            phase_name = re.sub(
                r"[^A-Za-z0-9_-]+", "_", phase
            )

            output = folder / f"{phase_name}_PPG_PRV_Features.csv"

            result.to_csv(output, index=False)
            all_rows.append(result)

            audit.append({
                "Source_File": str(path),
                "Participant": participant,
                "Phase": phase,
                "Sampling_Rate_Hz": fs,
                "Windows": len(result),
                "Status": "OK",
                "Output_File": str(output)
            })

        except Exception as exc:
            print(f"FAILED: {exc}")

            participant, phase = participant_phase(path)

            failed.append({
                "Source_File": str(path),
                "Error": str(exc)
            })

            audit.append({
                "Source_File": str(path),
                "Participant": participant,
                "Phase": phase,
                "Status": "FAILED",
                "Message": str(exc)
            })

    pd.DataFrame(audit).to_csv(
        OUTPUT_FOLDER / "PPG_FILE_AUDIT.csv",
        index=False
    )

    pd.DataFrame(failed).to_csv(
        OUTPUT_FOLDER / "PPG_FAILED_FILES.csv",
        index=False
    )

    if all_rows:
        combined = pd.concat(
            all_rows, ignore_index=True, sort=False
        )

        combined.to_csv(
            OUTPUT_FOLDER / "PPG_ALL_PARTICIPANTS_ALL_PHASES.csv",
            index=False
        )

        print("\n========== SUMMARY ==========")
        print("Successful files:", len(all_rows))
        print("Failed files:", len(failed))
        print("Total windows:", len(combined))

        print("\nParticipants:")
        print(combined["Participant"].nunique())

        print("\nWindows per participant and phase:")
        print(combined.groupby(["Participant", "Phase"]).size())

        print("\nProcessing status:")
        print(combined["Processing_Status"].value_counts())

    else:
        print("No files processed successfully.")

    print(f"\nOutput folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


