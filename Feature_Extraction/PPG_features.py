from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks, peak_prominences
warnings.filterwarnings("ignore")
INPUT_FOLDER = Path(r"D:\automate\EDA_Phase")
OUTPUT_FOLDER = Path(r"D:\EDA_DATA\Features_PPG")
WINDOW_SEC = 60
OVERLAP_SEC = 45
STEP_SEC = WINDOW_SEC - OVERLAP_SEC
FIXED_FS = None
MIN_VALID_IBI = 5
MAX_MISSING_FRACTION = 0.50
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
HRV_COLUMNS = """
HRV_MeanNN HRV_SDNN HRV_SDANN1 HRV_SDNNI1
HRV_SDANN2 HRV_SDNNI2 HRV_SDANN5 HRV_SDNNI5
HRV_RMSSD HRV_SDSD HRV_CVNN HRV_CVSD HRV_MedianNN
HRV_MadNN HRV_MCVNN HRV_IQRNN HRV_SDRMSSD
HRV_Prc20NN HRV_Prc80NN HRV_pNN50 HRV_pNN20
HRV_MinNN HRV_MaxNN HRV_HTI HRV_TINN
HRV_ULF HRV_VLF HRV_LF HRV_HF HRV_VHF HRV_TP
HRV_LFHF HRV_LFn HRV_HFn HRV_LnHF
HRV_SD1 HRV_SD2 HRV_SD1SD2 HRV_S HRV_CSI
HRV_CVI HRV_CSI_Modified HRV_PIP HRV_IALS
HRV_PSS HRV_PAS HRV_GI HRV_SI HRV_AI HRV_PI
HRV_C1d HRV_C1a HRV_SD1d HRV_SD1a
HRV_C2d HRV_C2a HRV_SD2d HRV_SD2a
HRV_Cd HRV_Ca HRV_SDNNd HRV_SDNNa
HRV_DFA_alpha1 HRV_MFDFA_alpha1_Width
HRV_MFDFA_alpha1_Peak HRV_MFDFA_alpha1_Mean
HRV_MFDFA_alpha1_Max HRV_MFDFA_alpha1_Delta
HRV_MFDFA_alpha1_Asymmetry HRV_MFDFA_alpha1_Fluctuation
HRV_MFDFA_alpha1_Increment HRV_ApEn HRV_SampEn
HRV_ShanEn HRV_FuzzyEn HRV_MSEn HRV_CMSEn
HRV_RCMSEn HRV_CD HRV_HFD HRV_KFD HRV_LZC
HRV_Symbolic_EqualProb4_0V HRV_Symbolic_EqualProb4_1V
HRV_Symbolic_EqualProb4_2LV HRV_Symbolic_EqualProb4_2UV
RecurrenceRate DiagRec
""".split()
EXTRA_COLUMNS = """
HRV_RMSSD_ms PPI_ms_std HR_std HR_mean HR_Acceleration
PPI_ms_mean PPG_Pulse_Peak_Count IBI_Raw_Count IBI_Valid_Count
PPG_Raw_Sample_Count PPG_Clean_Sample_Count PPG_Clean_Mean
PPG_Clean_SD PPG_Peak_Rate_per_sec
PPG_Heart_Rate_Min_bpm PPG_Heart_Rate_Max_bpm
PPG_Amplitude_mean PPG_Amplitude_std
PPG_RiseTime_s_mean PPG_RiseTime_s_std
PPG_FallTime_s_mean PPG_FallTime_s_std
PPG_PulseDuration_s_mean PPG_PulseDuration_s_std
PPG_DownstrokeSlope_std PPG_FirstDerivative_std
PPG_SecondDerivative_std
""".split()
META_COLUMNS = [
    "Participant",
    "Phase",
    "Source_File",
    "Window_Index",
    "window_start_sec",
    "window_end_sec",
    "Window_Start",
    "Window_End",
    "Sampling_Rate_Hz",
    "Window_Duration_s",
    "Missing_Data_Percent",
    "Processing_Status",
    "Processing_Message"
]
ALL_COLUMNS = META_COLUMNS + HRV_COLUMNS + EXTRA_COLUMNS
def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
def find_timestamp_column(df):
    preferred = [
        "timestamp", "datetime", "time", "date_time",
        "date time", "utc time"
    ]
    lookup = {normalize_name(c): c for c in df.columns}
    for name in preferred:
        key = normalize_name(name)
        if key in lookup:
            return lookup[key]
    for c in df.columns:
        n = normalize_name(c)
        if "timestamp" in n or n == "datetime":
            return c
    return None
def find_ppg_column(df, timestamp_col):
    for col in df.columns:
        normalized = normalize_name(col)
        if "_ppg_" in str(col).lower() or (
            normalized.endswith("ppga13cal")
            or ("ppg" in normalized and "cal" in normalized)
        ):
            if col == timestamp_col:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() > 10:
                return col
    return None
def infer_participant_and_phase(file_path):
    relative = file_path.relative_to(INPUT_FOLDER)
    participant = relative.parts[0] if len(relative.parts) > 1 else "Unknown"
    name = file_path.stem.lower()
    if "phase_summary" in name or name.endswith("_summary"):
        phase = "Summary"
    elif "baseline" in name:
        phase = "Baseline"
    elif "speech" in name and ("rest" in name or "stress" in name):
        phase = "SpeechRest"
    elif "2backpart1" in name or "2back_part1" in name or "2back1" in name:
        phase = "2Backpart1"
    elif "2backpart2" in name or "2back_part2" in name or "2back2" in name:
        phase = "2Backpart2"
    elif "2back" in name:
        phase = "2Back"
    else:
        phase = file_path.stem
    return participant, phase
def estimate_sampling_rate(timestamps):
    diffs = timestamps.diff().dt.total_seconds().dropna()
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan
    median_dt = float(diffs.median())
    if median_dt <= 0:
        return np.nan
    return 1.0 / median_dt
def load_recording(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    lower_name = file_path.name.lower()
    if (
        lower_name.endswith("_features.csv")
        or "phase_summary" in lower_name
        or lower_name in {
            "ppg_all_participants_all_phases.csv",
            "ppg_file_audit.csv",
            "ppg_failed_files.csv"
        }
    ):
        raise ValueError("Skipped generated feature/summary CSV")
    timestamp_col = find_timestamp_column(df)
    if timestamp_col is None:
        raise ValueError("Timestamp column was not found")
    ppg_col = find_ppg_column(df, timestamp_col)
    if ppg_col is None:
        raise ValueError("PPG signal column was not found")
    timestamps = pd.to_datetime(
        df[timestamp_col], errors="coerce", utc=True
    )
    ppg = pd.to_numeric(df[ppg_col], errors="coerce")
    valid = timestamps.notna() & ppg.notna()
    timestamps = timestamps[valid].reset_index(drop=True)
    ppg = ppg[valid].reset_index(drop=True)
    if len(ppg) < 100:
        raise ValueError("Too few valid PPG samples")
    order = np.argsort(timestamps.astype("int64").to_numpy())
    timestamps = timestamps.iloc[order].reset_index(drop=True)
    ppg = ppg.iloc[order].reset_index(drop=True)
    temp = pd.DataFrame({"timestamp": timestamps, "ppg": ppg})
    temp = temp.groupby("timestamp", as_index=False)["ppg"].mean()
    timestamps = temp["timestamp"]
    ppg = temp["ppg"].astype(float)
    fs_estimated = estimate_sampling_rate(timestamps)
    fs = FIXED_FS if FIXED_FS is not None else fs_estimated
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Could not estimate a valid sampling rate")
    start_ns = timestamps.iloc[0].value
    end_ns = timestamps.iloc[-1].value
    duration = (end_ns - start_ns) / 1e9
    if duration <= 0:
        raise ValueError("Invalid recording duration")
    n_samples = int(np.floor(duration * fs)) + 1
    if n_samples < 100:
        raise ValueError("Too few samples after time-grid reconstruction")
    grid_seconds = np.arange(n_samples) / fs
    original_seconds = (
        timestamps.astype("int64").to_numpy() - start_ns
    ) / 1e9
    signal = np.interp(
        grid_seconds, original_seconds, ppg.to_numpy()
    )
    original_dt = np.diff(original_seconds)
    typical_dt = np.median(original_dt[original_dt > 0])
    if not np.isfinite(typical_dt) or typical_dt <= 0:
        typical_dt = 1.0 / fs
    gap_limit = max(2.5 / fs, typical_dt * 2.5)
    gap_seconds = original_dt[original_dt > gap_limit].sum()
    missing_fraction = min(1.0, gap_seconds / duration)
    return {
        "signal": signal,
        "fs": float(fs),
        "start_timestamp": timestamps.iloc[0],
        "duration": duration,
        "missing_fraction": missing_fraction,
        "ppg_column": ppg_col,
        "raw_count": len(ppg),
    }
def blank_features():
    return {column: np.nan for column in HRV_COLUMNS + EXTRA_COLUMNS}
def copy_hrv_result(destination, result):
    if result is None or not isinstance(result, pd.DataFrame) or result.empty:
        return
    for col in result.columns:
        if col in destination:
            values = pd.to_numeric(result[col], errors="coerce")
            if len(values) and pd.notna(values.iloc[0]):
                destination[col] = float(values.iloc[0])
def safe_hrv_features(peaks, fs, features):
    peak_info = {"PPG_Peaks": np.asarray(peaks, dtype=int)}
    try:
        result = nk.hrv_time(
            peak_info, sampling_rate=fs, show=False
        )
        copy_hrv_result(features, result)
    except Exception:
        pass
    try:
        result = nk.hrv_frequency(
            peak_info, sampling_rate=fs, show=False,
            normalize=True, order_criteria=True
        )
        copy_hrv_result(features, result)
    except Exception:
        pass
    try:
        result = nk.hrv_nonlinear(
            peak_info, sampling_rate=fs, show=False
        )
        copy_hrv_result(features, result)
    except Exception:
        pass
    try:
        if hasattr(nk, "hrv_rqa"):
            result = nk.hrv_rqa(
                peak_info, sampling_rate=fs, show=False
            )
            copy_hrv_result(features, result)
    except Exception:
        pass
def extract_window_features(raw_window, fs):
    features = blank_features()
    raw_window = np.asarray(raw_window, dtype=float)
    raw_count = len(raw_window)
    features["PPG_Raw_Sample_Count"] = raw_count
    features["IBI_Raw_Count"] = 0
    features["IBI_Valid_Count"] = 0
    features["PPG_Pulse_Peak_Count"] = 0
    if raw_count < int(WINDOW_SEC * fs * 0.8):
        return features, "FAILED", "Window has too few samples"
    missing = ~np.isfinite(raw_window)
    missing_fraction = float(missing.mean())
    if missing_fraction > MAX_MISSING_FRACTION:
        return features, "FAILED", "Too much missing signal data"
    valid_indices = np.flatnonzero(~missing)
    if len(valid_indices) < 10:
        return features, "FAILED", "Insufficient valid PPG samples"
    filled = np.interp(
        np.arange(raw_count), valid_indices, raw_window[valid_indices]
    )
    try:
        cleaned = nk.ppg_clean(
            filled, sampling_rate=fs, method="elgendi"
        )
    except Exception as e:
        return features, "FAILED", f"PPG cleaning failed: {e}"
    try:
        _, peak_info = nk.ppg_peaks(
            cleaned, sampling_rate=fs, method="elgendi", show=False
        )
        peaks = np.asarray(peak_info.get("PPG_Peaks", []), dtype=int)
    except Exception as e:
        return features, "FAILED", f"PPG peak detection failed: {e}"
    peaks = peaks[(peaks >= 0) & (peaks < len(cleaned))]
    peaks = np.unique(peaks)
    features["PPG_Clean_Sample_Count"] = len(cleaned)
    features["PPG_Clean_Mean"] = float(np.mean(cleaned))
    features["PPG_Clean_SD"] = float(np.std(cleaned, ddof=1))
    features["PPG_Pulse_Peak_Count"] = int(len(peaks))
    features["IBI_Raw_Count"] = max(0, len(peaks) - 1)
    features["Missing_Data_Percent"] = missing_fraction * 100
    if len(peaks) >= 2:
        ibi_seconds = np.diff(peaks) / fs
        valid_ibi = ibi_seconds[
            (ibi_seconds >= 0.30) & (ibi_seconds <= 2.00)
        ]
        features["IBI_Valid_Count"] = int(len(valid_ibi))
        features["PPI_ms_mean"] = (
            float(np.mean(valid_ibi) * 1000)
            if len(valid_ibi) else np.nan
        )
        features["PPI_ms_std"] = (
            float(np.std(valid_ibi, ddof=1) * 1000)
            if len(valid_ibi) > 1 else np.nan
        )
        if len(valid_ibi):
            heart_rate = 60.0 / valid_ibi
            features["HR_mean"] = float(np.mean(heart_rate))
            features["HR_std"] = (
                float(np.std(heart_rate, ddof=1))
                if len(heart_rate) > 1 else np.nan
            )
            features["PPG_Heart_Rate_Min_bpm"] = float(np.min(heart_rate))
            features["PPG_Heart_Rate_Max_bpm"] = float(np.max(heart_rate))
            features["PPG_Peak_Rate_per_sec"] = len(peaks) / (len(cleaned) / fs)
            if len(heart_rate) > 1:
                features["HR_Acceleration"] = float(
                    np.mean(np.diff(heart_rate))
                )
        if len(valid_ibi) >= MIN_VALID_IBI:
            valid_peak_positions = [peaks[0]]
            for i, interval in enumerate(ibi_seconds):
                if 0.30 <= interval <= 2.00:
                    valid_peak_positions.append(peaks[i + 1])
            valid_peak_positions = np.asarray(
                valid_peak_positions, dtype=int
            )
            if len(valid_peak_positions) >= MIN_VALID_IBI:
                safe_hrv_features(valid_peak_positions, fs, features)
        if len(valid_ibi) > 1:
            diff_ibi_ms = np.diff(valid_ibi) * 1000
            features["HRV_RMSSD_ms"] = float(
                np.sqrt(np.mean(diff_ibi_ms ** 2))
            )
    if len(peaks) >= 2:
        troughs = []
        amplitudes = []
        rise_times = []
        fall_times = []
        durations = []
        slopes = []
        for i, peak in enumerate(peaks):
            previous_peak = peaks[i - 1] if i > 0 else 0
            next_peak = peaks[i + 1] if i + 1 < len(peaks) else len(cleaned) - 1
            if peak <= previous_peak or next_peak <= peak:
                continue
            segment_before = cleaned[previous_peak:peak + 1]
            if len(segment_before) < 3:
                continue
            trough = previous_peak + int(np.argmin(segment_before))
            if trough >= peak:
                continue
            amplitude = cleaned[peak] - cleaned[trough]
            rise_time = (peak - trough) / fs
            fall_time = (next_peak - peak) / fs
            duration = (next_peak - trough) / fs
            if not np.isfinite(amplitude):
                continue
            troughs.append(trough)
            amplitudes.append(float(amplitude))
            rise_times.append(float(rise_time))
            fall_times.append(float(fall_time))
            durations.append(float(duration))
            if fall_time > 0:
                slopes.append(float(
                    (cleaned[next_peak] - cleaned[peak]) / fall_time
                ))
        def mean_or_nan(values):
            return float(np.mean(values)) if len(values) else np.nan
        def std_or_nan(values):
            return (
                float(np.std(values, ddof=1))
                if len(values) > 1 else np.nan
            )
        features["PPG_Amplitude_mean"] = mean_or_nan(amplitudes)
        features["PPG_Amplitude_std"] = std_or_nan(amplitudes)
        features["PPG_RiseTime_s_mean"] = mean_or_nan(rise_times)
        features["PPG_RiseTime_s_std"] = std_or_nan(rise_times)
        features["PPG_FallTime_s_mean"] = mean_or_nan(fall_times)
        features["PPG_FallTime_s_std"] = std_or_nan(fall_times)
        features["PPG_PulseDuration_s_mean"] = mean_or_nan(durations)
        features["PPG_PulseDuration_s_std"] = std_or_nan(durations)
        features["PPG_DownstrokeSlope_std"] = std_or_nan(slopes)
    first_derivative = np.diff(cleaned) * fs
    second_derivative = np.diff(first_derivative) * fs
    features["PPG_FirstDerivative_std"] = (
        float(np.std(first_derivative, ddof=1))
        if len(first_derivative) > 1 else np.nan
    )
    features["PPG_SecondDerivative_std"] = (
        float(np.std(second_derivative, ddof=1))
        if len(second_derivative) > 1 else np.nan
    )
    if len(peaks) < 2:
        return features, "WARNING", "Too few detected pulse peaks"
    return features, "OK", ""
def process_file(file_path):
    participant, phase = infer_participant_and_phase(file_path)
    recording = load_recording(file_path)
    signal = recording["signal"]
    fs = recording["fs"]
    window_samples = int(round(WINDOW_SEC * fs))
    step_samples = int(round(STEP_SEC * fs))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Invalid window or step size")
    rows = []
    window_index = 0
    for start in range(0, len(signal), step_samples):
        end = start + window_samples
        if end > len(signal):
            available = len(signal) - start
            if available < int(window_samples * 0.80):
                break
            window = signal[start:]
        else:
            window = signal[start:end]
        window_index = len(rows)
        window_start_sec = start / fs
        window_end_sec = window_start_sec + len(window) / fs
        actual_duration = len(window) / fs
        features, status, message = extract_window_features(
            window, fs
        )
        start_time = (
            recording["start_timestamp"]
            + pd.to_timedelta(window_start_sec, unit="s")
        )
        end_time = (
            recording["start_timestamp"]
            + pd.to_timedelta(window_end_sec, unit="s")
        )
        row = {col: np.nan for col in ALL_COLUMNS}
        row.update(features)
        row.update({
            "Participant": participant,
            "Phase": phase,
            "Source_File": file_path.name,
            "Window_Index": window_index,
            "window_start_sec": round(window_start_sec, 6),
            "window_end_sec": round(window_end_sec, 6),
            "Window_Start": start_time.isoformat(),
            "Window_End": end_time.isoformat(),
            "Sampling_Rate_Hz": fs,
            "Window_Duration_s": actual_duration,
            "Missing_Data_Percent": (
                recording["missing_fraction"] * 100
            ),
            "Processing_Status": status,
            "Processing_Message": message,
        })
        rows.append(row)
        if end >= len(signal):
            break
    result = pd.DataFrame(rows).reindex(columns=ALL_COLUMNS)
    return result, participant, phase, recording["ppg_column"], fs
def main():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {INPUT_FOLDER}"
        )
    all_csv_files = sorted(INPUT_FOLDER.rglob("*.csv"))
    output_resolved = OUTPUT_FOLDER.resolve()
    candidate_files = []
    for path in all_csv_files:
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        lower = path.name.lower()
        if (
            lower.endswith("_features.csv")
            or "phase_summary" in lower
            or lower in {
                "ppg_all_participants_all_phases.csv",
                "ppg_file_audit.csv",
                "ppg_failed_files.csv"
            }
        ):
            continue
        candidate_files.append(path)
    if not candidate_files:
        print("No input CSV files found.")
        return
    all_results = []
    audit_rows = []
    failed_rows = []
    for index, file_path in enumerate(candidate_files, start=1):
        print(f"[{index}/{len(candidate_files)}] Processing: {file_path}")
        try:
            result, participant, phase, ppg_col, fs = process_file(file_path)
            participant_folder = OUTPUT_FOLDER / participant
            participant_folder.mkdir(parents=True, exist_ok=True)
            safe_phase = re.sub(r"[^A-Za-z0-9_-]+", "_", phase)
            phase_file = (
                participant_folder
                / f"{safe_phase}_PPG_HRV_Features.csv"
            )
            if phase_file.exists():
                previous = pd.read_csv(phase_file)
                result = pd.concat(
                    [previous, result], ignore_index=True
                )
            result.to_csv(phase_file, index=False)
            all_results.append(result)
            audit_rows.append({
                "Participant": participant,
                "Phase": phase,
                "Source_File": str(file_path),
                "PPG_Column": ppg_col,
                "Sampling_Rate_Hz": fs,
                "Windows_Generated": len(result),
                "Output_File": str(phase_file),
                "Status": "OK",
                "Message": "",
            })
            print(
                f"   Saved {len(result)} windows -> {phase_file}"
            )
        except Exception as e:
            print(f"   FAILED: {e}")
            failed_rows.append({
                "Source_File": str(file_path),
                "Error": str(e),
            })
            audit_rows.append({
                "Participant": infer_participant_and_phase(file_path)[0],
                "Phase": infer_participant_and_phase(file_path)[1],
                "Source_File": str(file_path),
                "PPG_Column": "",
                "Sampling_Rate_Hz": np.nan,
                "Windows_Generated": 0,
                "Output_File": "",
                "Status": "FAILED",
                "Message": str(e),
            })
        pd.DataFrame(audit_rows).to_csv(
            OUTPUT_FOLDER / "PPG_FILE_AUDIT.csv", index=False
        )
        pd.DataFrame(failed_rows).to_csv(
            OUTPUT_FOLDER / "PPG_FAILED_FILES.csv", index=False
        )
        if all_results:
            pd.concat(all_results, ignore_index=True).reindex(
                columns=ALL_COLUMNS
            ).to_csv(
                OUTPUT_FOLDER / "PPG_ALL_PARTICIPANTS_ALL_PHASES.csv",
                index=False
            )
    print("\n================ FINAL SUMMARY ================")
    print(f"Input files discovered: {len(candidate_files)}")
    print(f"Files processed successfully: {len(all_results)}")
    print(f"Files failed: {len(failed_rows)}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"Total windows in combined output: {len(combined)}")
        print("\nWindows by participant and phase:")
        print(
            combined.groupby(["Participant", "Phase"])
            .size()
            .to_string()
        )
    print("================================================")
if __name__ == "__main__":
    main()
