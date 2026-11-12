from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from openpyxl import load_workbook


# ============================================================
# 1. EDIT THESE PATHS
# ============================================================

# Put all EDA ZIP/CSV files anywhere under this folder.
# The script searches recursively.
#
# Example:
# D:\AryaStudy\EDA_Data\
#   Arya_Session1_P1_Calibrated_SD.zip
#   Arya_Session1_P2_Calibrated_SD.zip
#   Arya_Session1_P3_Calibrated_SD.zip
#
# OR:
# D:\AryaStudy\EDA_Data\
#   P1\Arya_Session1_P1_Calibrated_SD.csv
#   P2\Arya_Session1_P2_Calibrated_SD.csv
#
EDA_ROOT = Path(r"E:\Cognitive[Arya]\Arya_study_data\EDA")

# Master timing file:
#   BaselineStart / BaselineEnd
#   SR_Start / SR_End
#   SpeechOrRest
MASTER_XLSX = Path(r"D:\automate\AryaDataMarking_copy.xlsx")

# 12-block timing workbook:
#   Block1 ... Block12
#   P_Date / 2BackStart / 2BackEnd
BLOCKS_XLSX = Path(r"D:\automate\Arya_Block1_to_Block12_Corrected.xlsx")

# Used only when a participant's corrected Block1...Block12 sequence is
# objectively inconsistent with the Master 2-back interval. The corrected
# workbook remains the primary source for every normal participant.
BLOCKS_FALLBACK_XLSX = None
# Output location.
OUTPUT_ROOT = Path(r"D:\automate\EDA_Phase")

# Can be overridden by targeted audit/repair runs.
GLOBAL_SUMMARY_FILENAME = (
    "all_participants_eda_extraction_summary_full_timestamps3.csv"
)


# ============================================================
# 2. OPTIONS
# ============================================================

# True = do not repeat a participant after it has completed successfully.
SKIP_COMPLETED_PARTICIPANTS = False

# Create a second, combined CSV for each participant containing
# Baseline + Block1...Block12 + Speech/Rest and metadata columns.
WRITE_PARTICIPANT_COMBINED_CSV = True

# Phase interval rule:
#     phase_start <= EDA timestamp <= phase_end
INCLUDE_END_TIMESTAMP = True

# If you want to test first:
# PARTICIPANTS_TO_PROCESS = ["P1"]
#
# For the final complete run:
PARTICIPANTS_TO_PROCESS = None


# ============================================================
# 3. ONLY THESE CORRECT PARTICIPANTS
# ============================================================

CORRECT_PARTICIPANTS = [
    "P8"
]


# ============================================================
# 4. EDA COLUMNS TO EXTRACT
# ============================================================

# The participant prefix is added automatically:
# P1_Timestamp_FormattedUnix_CAL
# P1_GSR_Range_CAL
# ...
EDA_COLUMN_SUFFIXES = [
    "Timestamp_FormattedUnix_CAL",
    "GSR_Range_CAL",
    "GSR_Skin_Conductance_CAL",
    "GSR_Skin_Resistance_CAL",
    "PPG_A13_CAL",
    "PPG_IBI_CAL",
    "PPGtoHR_CAL",
    "Pressure_BMP280_CAL",
    "Temperature_BMP280_CAL",
]

# Some older recordings (currently P165) do not contain these two derived
# PPG columns. Their output columns are still created and left blank rather
# than rejecting the participant's otherwise valid EDA source.
OPTIONAL_EDA_COLUMN_SUFFIXES = {
    "PPG_IBI_CAL",
    "PPGtoHR_CAL",
}

# Units in the uploaded sample:
# Timestamp                 yyyy/mm/dd hh:mm:ss.000
# GSR_Range                 no_units
# GSR_Skin_Conductance      uS
# GSR_Skin_Resistance       kOhms
# PPG_A13                   mV
# PPG_IBI                   ms
# PPGtoHR                   BPM
# Pressure_BMP280           kPa
# Temperature_BMP280        Degrees Celsius


# ============================================================
# 5. DATA STRUCTURES
# ============================================================

@dataclass
class PhaseWindow:
    phase: str
    block: int | None
    start: datetime
    end: datetime
    condition: str = ""

    @property
    def label(self) -> str:
        if self.block is not None:
            return f"2Back_Block{self.block:02d}"
        return self.phase


@dataclass
class EdaSource:
    path: Path
    zip_member: str | None = None

    def display_name(self) -> str:
        if self.zip_member:
            return f"{self.path} :: {self.zip_member}"
        return str(self.path)


# ============================================================
# 6. GENERAL HELPERS
# ============================================================

def normalize_pid(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    m = re.fullmatch(r"P\s*(\d+)", text)
    if not m:
        return None
    return f"P{int(m.group(1))}"


def pid_number(pid: str) -> int:
    m = re.fullmatch(r"P(\d+)", pid, re.I)
    return int(m.group(1)) if m else 10**9


def parse_excel_date(value) -> date | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    return None


def parse_excel_time(value) -> time | None:
    """
    Accepts:
      Excel time object
      Excel numeric day fraction
      timedelta
      strings such as 12:06:27 or 12:06:27.652150
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.time()

    if isinstance(value, time):
        return value

    if isinstance(value, timedelta):
        seconds = value.total_seconds()
        if seconds < 0:
            return None
        seconds %= 86400
        whole = int(seconds)
        microseconds = int(round((seconds - whole) * 1_000_000))
        h, rem = divmod(whole, 3600)
        m, s = divmod(rem, 60)
        return time(h, m, s, microseconds)

    # Excel time stored as a fraction of 1 day.
    if isinstance(value, (int, float)):
        fraction = float(value) % 1.0
        total_microseconds = round(fraction * 86400 * 1_000_000)
        total_microseconds %= 86400 * 1_000_000

        h, rem = divmod(total_microseconds, 3600 * 1_000_000)
        m, rem = divmod(rem, 60 * 1_000_000)
        s, us = divmod(rem, 1_000_000)
        return time(int(h), int(m), int(s), int(us))

    text = str(value).strip()
    if not text:
        return None

    # Sometimes the date is included; keep only time if parseable.
    for fmt in (
        "%H:%M:%S.%f",
        "%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass

    return None


def combine_date_and_time(d: date, t: time) -> datetime:
    return datetime.combine(d, t)


def make_interval(
    d: date,
    start_time: time,
    end_time: time,
) -> tuple[datetime, datetime]:
    """
    If end clock time is earlier than start clock time,
    treat the end as next day (midnight crossing).
    """
    start_dt = combine_date_and_time(d, start_time)
    end_dt = combine_date_and_time(d, end_time)

    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    return start_dt, end_dt


def parse_eda_timestamp(text: str) -> datetime | None:
    """Parse an EDA value that already contains a complete date and time."""
    text = str(text).strip()

    if not text:
        return None

    # Uploaded sample:
    # 2024/10/26 11:58:07.318
    for fmt in (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d_%H:%M:%S.%f",
        "%Y-%m-%d_%H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    return None


def parse_minute_second_timestamp(text: str) -> float | None:
    """
    Parse the truncated Shimmer values used by some participants.

    Examples:
      58:26.5 -> 58 minutes and 26.5 seconds within an unknown hour
      03:54.0 -> 3 minutes and 54 seconds within an unknown hour

    The missing date/hour is reconstructed by EdaTimestampDecoder from
    the participant's master/corrected phase timeline. The source value
    itself is never rewritten.
    """
    match = re.fullmatch(
        r"\s*(\d{1,2}):([0-5]?\d(?:\.\d+)?)\s*",
        str(text),
    )
    if not match:
        return None

    minute = int(match.group(1))
    second = float(match.group(2))

    if minute > 59 or not (0.0 <= second < 60.0):
        return None

    return minute * 60.0 + second


def format_output_timestamp(value: datetime) -> str:
    """Return the complete timestamp format required in every phase CSV."""
    return value.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]


class EdaTimestampDecoder:
    """
    Decode either full timestamps or minute:second clock fragments.

    For a fragment, the nearest plausible occurrence to the first phase
    is selected. Subsequent 59 -> 00 transitions advance the hour. This
    lets extraction use absolute master-timeline datetimes without
    converting or modifying the participant CSV first.
    """

    def __init__(self, windows: list[PhaseWindow]):
        self.reference = min(window.start for window in windows)
        self.mode = "UNKNOWN"
        self.anchor_hour: datetime | None = None
        self.previous_clock_seconds: float | None = None
        self.hour_rollovers = 0

    def _set_fragment_anchor(self, clock_seconds: float) -> None:
        reference_hour = self.reference.replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        candidates = [
            reference_hour
            + timedelta(hours=hour_offset, seconds=clock_seconds)
            for hour_offset in range(-2, 3)
        ]

        # Prefer the closest clock occurrence. On an exact tie, prefer a
        # recording start at/before the first phase rather than after it.
        first_timestamp = min(
            candidates,
            key=lambda candidate: (
                abs((candidate - self.reference).total_seconds()),
                candidate > self.reference,
            ),
        )
        self.anchor_hour = first_timestamp - timedelta(
            seconds=clock_seconds
        )

    def parse(self, value: str) -> datetime | None:
        absolute = parse_eda_timestamp(value)
        if absolute is not None:
            if self.mode == "UNKNOWN":
                self.mode = "ABSOLUTE"
            return absolute

        clock_seconds = parse_minute_second_timestamp(value)
        if clock_seconds is None:
            return None

        if self.anchor_hour is None:
            self._set_fragment_anchor(clock_seconds)
            self.mode = "MINUTE_SECOND_RECONSTRUCTED"

        if (
            self.previous_clock_seconds is not None
            and self.previous_clock_seconds - clock_seconds > 1800.0
        ):
            self.hour_rollovers += 1

        self.previous_clock_seconds = clock_seconds

        return (
            self.anchor_hour
            + timedelta(hours=self.hour_rollovers, seconds=clock_seconds)
        )

    @property
    def anchor_text(self) -> str:
        if self.anchor_hour is None:
            return ""
        return self.anchor_hour.strftime("%Y-%m-%d %H:00:00")


# ============================================================
# 7. READ TIMING EXCEL FILES
# ============================================================

def sheet_rows_by_pid(ws) -> dict[str, dict]:
    iterator = ws.iter_rows(values_only=True)

    try:
        headers_raw = next(iterator)
    except StopIteration:
        return {}

    headers = [
        str(v).strip() if v is not None else ""
        for v in headers_raw
    ]

    pid_index = None
    for candidate in ("ParticipantId", "ParticipantID"):
        if candidate in headers:
            pid_index = headers.index(candidate)
            break

    if pid_index is None:
        raise ValueError(
            f"No ParticipantId/ParticipantID column in sheet '{ws.title}'. "
            f"Headers: {headers}"
        )

    records = {}

    for row in iterator:
        if not row or pid_index >= len(row):
            continue

        pid = normalize_pid(row[pid_index])
        if not pid:
            continue

        record = {}
        for i, header in enumerate(headers):
            if header:
                record[header] = row[i] if i < len(row) else None

        records[pid] = record

    return records


def block_sequence_matches_master(
    pid: str,
    master_row: dict,
    block_tables: dict[int, dict[str, dict]],
    tolerance_seconds: float = 120.0,
) -> bool:
    """Check the per-block sequence against Master 2BackStart/2BackEnd."""
    first_row = block_tables.get(1, {}).get(pid)
    last_row = block_tables.get(12, {}).get(pid)
    if not first_row or not last_row:
        return False

    experiment_date = parse_excel_date(first_row.get("P_Date"))
    first_date = parse_excel_date(first_row.get("P_Date"))
    last_date = parse_excel_date(last_row.get("P_Date"))
    first_time = parse_excel_time(first_row.get("2BackStart"))
    last_time = parse_excel_time(last_row.get("2BackEnd"))
    master_start = parse_excel_time(master_row.get("2BackStart"))
    master_end = parse_excel_time(master_row.get("2BackEnd"))

    if not all(
        (
            experiment_date,
            first_date,
            last_date,
            first_time,
            last_time,
            master_start,
            master_end,
        )
    ):
        return False

    block_start = combine_date_and_time(first_date, first_time)
    block_end = combine_date_and_time(last_date, last_time)
    master_start_dt, master_end_dt = make_interval(
        experiment_date,
        master_start,
        master_end,
    )

    return (
        abs((block_start - master_start_dt).total_seconds())
        <= tolerance_seconds
        and abs((block_end - master_end_dt).total_seconds())
        <= tolerance_seconds
    )


def load_timing_tables():
    if not MASTER_XLSX.exists():
        raise FileNotFoundError(f"MASTER_XLSX not found:\n{MASTER_XLSX}")

    if not BLOCKS_XLSX.exists():
        raise FileNotFoundError(f"BLOCKS_XLSX not found:\n{BLOCKS_XLSX}")

    master_wb = load_workbook(
        MASTER_XLSX,
        read_only=True,
        data_only=True,
    )

    blocks_wb = load_workbook(
        BLOCKS_XLSX,
        read_only=True,
        data_only=True,
    )

    if "Master" not in master_wb.sheetnames:
        raise ValueError(
            f"'Master' sheet not found in {MASTER_XLSX.name}. "
            f"Available sheets: {master_wb.sheetnames}"
        )

    master = sheet_rows_by_pid(master_wb["Master"])

    block_tables = {}

    for block in range(1, 13):
        sheet_name = f"Block{block}"

        if sheet_name not in blocks_wb.sheetnames:
            raise ValueError(
                f"'{sheet_name}' missing from {BLOCKS_XLSX.name}"
            )

        block_tables[block] = sheet_rows_by_pid(
            blocks_wb[sheet_name]
        )

    inconsistent = [
        pid
        for pid, master_row in master.items()
        if not block_sequence_matches_master(
            pid,
            master_row,
            block_tables,
        )
    ]

    # BLOCKS_FALLBACK_XLSX is optional; do not call .exists() when it is None.
    if (
        inconsistent
        and BLOCKS_FALLBACK_XLSX is not None
        and BLOCKS_FALLBACK_XLSX.exists()
    ):
        fallback_wb = load_workbook(
            BLOCKS_FALLBACK_XLSX,
            read_only=True,
            data_only=True,
        )
        fallback_tables = {
            block: sheet_rows_by_pid(fallback_wb[f"Block{block}"])
            for block in range(1, 13)
            if f"Block{block}" in fallback_wb.sheetnames
        }

        for pid in inconsistent:
            fallback_candidate = {
                block: fallback_tables.get(block, {}).get(pid)
                for block in range(1, 13)
            }
            if not all(fallback_candidate.values()):
                continue

            candidate_tables = {
                block: {pid: fallback_candidate[block]}
                for block in range(1, 13)
            }
            if not block_sequence_matches_master(
                pid,
                master[pid],
                candidate_tables,
            ):
                continue

            for block in range(1, 13):
                block_tables[block][pid] = fallback_candidate[block]

            print(
                f"WARNING: {pid} corrected block sequence conflicts "
                f"with Master; using validated fallback from "
                f"{BLOCKS_FALLBACK_XLSX.name}."
            )

    return master, block_tables


def participant_experiment_date(
    pid: str,
    block_tables: dict[int, dict[str, dict]],
) -> date | None:
    """
    Baseline/SpeechRest Master sheet has clock time but no P_Date.
    We take the experiment date from the participant's block sheet.
    """
    for block in range(1, 13):
        row = block_tables[block].get(pid)
        if not row:
            continue

        d = parse_excel_date(row.get("P_Date"))
        if d:
            return d

    return None


def build_phase_windows(
    pid: str,
    master: dict[str, dict],
    block_tables: dict[int, dict[str, dict]],
) -> list[PhaseWindow]:

    master_row = master.get(pid)

    if not master_row:
        raise ValueError(f"{pid}: missing from Master sheet")

    experiment_date = participant_experiment_date(
        pid,
        block_tables,
    )

    if not experiment_date:
        raise ValueError(
            f"{pid}: could not determine experiment P_Date "
            f"from Block1...Block12"
        )

    condition = str(
        master_row.get("SpeechOrRest") or ""
    ).strip()

    windows: list[PhaseWindow] = []

    # --------------------------------------------------------
    # BASELINE
    # Use ACTUAL experiment timestamps, not P_Vid_* offsets.
    # --------------------------------------------------------
    baseline_start = parse_excel_time(
        master_row.get("BaselineStart")
    )
    baseline_end = parse_excel_time(
        master_row.get("BaselineEnd")
    )

    if baseline_start and baseline_end:
        start_dt, end_dt = make_interval(
            experiment_date,
            baseline_start,
            baseline_end,
        )
        windows.append(
            PhaseWindow(
                phase="Baseline",
                block=None,
                start=start_dt,
                end=end_dt,
                condition=condition,
            )
        )

    # --------------------------------------------------------
    # 2-BACK BLOCK 1 -> 12
    # Use P_Date + 2BackStart / 2BackEnd.
    # --------------------------------------------------------
    for block in range(1, 13):
        row = block_tables[block].get(pid)

        if not row:
            print(f"WARNING: {pid} missing from Block{block}")
            continue

        block_date = (
            parse_excel_date(row.get("P_Date"))
            or experiment_date
        )
        start_t = parse_excel_time(row.get("2BackStart"))
        end_t = parse_excel_time(row.get("2BackEnd"))

        if not start_t or not end_t:
            print(
                f"WARNING: {pid} Block{block} has missing/invalid "
                f"2BackStart or 2BackEnd"
            )
            continue

        start_dt, end_dt = make_interval(
            block_date,
            start_t,
            end_t,
        )

        windows.append(
            PhaseWindow(
                phase="2Back",
                block=block,
                start=start_dt,
                end=end_dt,
                condition=condition,
            )
        )

    # --------------------------------------------------------
    # SPEECH / REST
    # Use ACTUAL experiment timestamps, not P_SR_* video offsets.
    # --------------------------------------------------------
    sr_start = parse_excel_time(master_row.get("SR_Start"))
    sr_end = parse_excel_time(master_row.get("SR_End"))

    if sr_start and sr_end:
        start_dt, end_dt = make_interval(
            experiment_date,
            sr_start,
            sr_end,
        )

        # If the speech/rest window belongs after midnight while
        # the final 2-back block was on the previous date, adjust.
        if windows:
            latest_before_sr = max(w.end for w in windows)
            if end_dt < latest_before_sr and (latest_before_sr - end_dt) > timedelta(hours=12):
                start_dt += timedelta(days=1)
                end_dt += timedelta(days=1)

        windows.append(
            PhaseWindow(
                phase="SpeechRest",
                block=None,
                start=start_dt,
                end=end_dt,
                condition=condition,
            )
        )

    windows.sort(key=lambda w: w.start)

    if not windows:
        raise ValueError(f"{pid}: no valid phase windows found")

    return windows


# ============================================================
# 8. FIND EDA ZIP / CSV SOURCE
# ============================================================

def filename_pid(path_text: str) -> str | None:
    """
    Safely extracts P1 from:
      Arya_Session1_P1_Calibrated_SD.zip
      P1_data.csv

    It will not confuse P1 with P10.
    """
    name = Path(path_text).name.upper()

    matches = re.findall(
        r"(?<![A-Z0-9])P(\d+)(?!\d)",
        name,
    )

    if not matches:
        # Common underscore-separated form:
        matches = re.findall(r"(?:^|[_\-\s])P(\d+)(?=[_\-\s.]|$)", name)

    if not matches:
        return None

    return f"P{int(matches[-1])}"


def discover_eda_sources(
    eda_root: Path,
) -> dict[str, list[EdaSource]]:

    if not eda_root.exists():
        raise FileNotFoundError(f"EDA_ROOT does not exist:\n{eda_root}")

    result: dict[str, list[EdaSource]] = {}

    for path in eda_root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".zip":
            outer_pid = filename_pid(path.name)

            try:
                with zipfile.ZipFile(path, "r") as z:
                    for member in z.namelist():
                        member_lower = member.lower()
                        if not member_lower.endswith((".csv", ".tsv", ".txt")):
                            continue

                        pid = filename_pid(member) or outer_pid

                        if pid:
                            result.setdefault(pid, []).append(
                                EdaSource(path=path, zip_member=member)
                            )
            except zipfile.BadZipFile:
                print(f"WARNING: bad ZIP skipped: {path}")

        elif suffix in {".csv", ".tsv", ".txt"}:
            pid = filename_pid(path.name)

            if pid:
                result.setdefault(pid, []).append(
                    EdaSource(path=path)
                )

    return result


@contextmanager
def open_eda_text(source: EdaSource):
    """
    Opens either:
      normal CSV/TSV
      CSV/TSV directly inside ZIP

    Nothing is fully loaded into RAM.
    """
    if source.zip_member is None:
        with source.path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as f:
            yield f
        return

    with zipfile.ZipFile(source.path, "r") as z:
        with z.open(source.zip_member, "r") as raw:
            with io.TextIOWrapper(
                raw,
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as text:
                yield text


def read_header_and_delimiter(text_file):
    """
    Handles Shimmer-style files such as:

      sep=\\t
      P1_Timestamp_FormattedUnix_CAL  P1_GSR_Range_CAL ...
      yyyy/mm/dd hh:mm:ss.000         no_units ...
    """
    first = text_file.readline()

    if not first:
        raise ValueError("EDA file is empty")

    # Shimmer exports may write either:
    #   sep=<TAB>
    # or:
    #   "sep=<TAB>"
    # so remove only the wrapping quotes/newline, not the TAB itself.
    first_no_newline = first.rstrip("\r\n")
    first_unquoted = first_no_newline

    if (
        len(first_unquoted) >= 2
        and first_unquoted[0] == '"'
        and first_unquoted[-1] == '"'
    ):
        first_unquoted = first_unquoted[1:-1]

    if first_unquoted.lower().startswith("sep="):
        raw_delimiter = first_unquoted[4:]

        if raw_delimiter == r"\t" or raw_delimiter == "\t":
            delimiter = "\t"
        elif raw_delimiter:
            delimiter = raw_delimiter[0]
        else:
            delimiter = "\t"

        header_line = text_file.readline()

    else:
        header_line = first
        delimiter = (
            "\t"
            if header_line.count("\t") >= header_line.count(",")
            else ","
        )

    header = next(
        csv.reader(
            [header_line],
            delimiter=delimiter,
        )
    )

    header = [str(x).strip() for x in header]

    # Remove trailing empty field caused by a trailing delimiter.
    while header and header[-1] == "":
        header.pop()

    return header, delimiter


def required_columns_for_pid(pid: str) -> list[str]:
    return [
        f"{pid}_{suffix}"
        for suffix in EDA_COLUMN_SUFFIXES
    ]


def source_matches_pid(
    source: EdaSource,
    pid: str,
) -> bool:
    try:
        with open_eda_text(source) as text:
            header, _ = read_header_and_delimiter(text)
            required = [
                f"{pid}_{suffix}"
                for suffix in EDA_COLUMN_SUFFIXES
                if suffix not in OPTIONAL_EDA_COLUMN_SUFFIXES
            ]
            return all(col in header for col in required)
    except Exception:
        return False


def select_source_for_pid(
    pid: str,
    source_index: dict[str, list[EdaSource]],
) -> EdaSource | None:

    candidates = source_index.get(pid, [])

    if not candidates:
        return None

    # ------------------------------------------------------------
    # Check which candidate files actually contain the required
    # columns for this participant.
    # ------------------------------------------------------------
    valid = [
        source
        for source in candidates
        if source_matches_pid(source, pid)
    ]

    # ------------------------------------------------------------
    # CASE 1: Exactly one valid source
    # ------------------------------------------------------------
    if len(valid) == 1:
        return valid[0]

    # ------------------------------------------------------------
    # CASE 2: Multiple valid sources
    # ------------------------------------------------------------
    if len(valid) > 1:

        # Direct CSV/TSV/TXT files have zip_member == None.
        # Files inside ZIP archives have zip_member set.
        direct_sources = [
            source
            for source in valid
            if source.zip_member is None
        ]

        # --------------------------------------------------------
        # Exactly one direct source:
        # Prefer it over the same file inside a ZIP.
        # --------------------------------------------------------
        if len(direct_sources) == 1:

            selected_source = direct_sources[0]

            print(
                f"Multiple valid sources found for {pid}."
            )
            print(
                "Preferring direct CSV over ZIP copy:"
            )
            print(
                f"    {selected_source.display_name()}"
            )

            return selected_source

        # --------------------------------------------------------
        # More than one direct CSV:
        # Cannot safely decide which one is correct.
        # --------------------------------------------------------
        if len(direct_sources) > 1:

            print(
                f"ERROR: {pid} has multiple direct EDA sources:"
            )

            for source in direct_sources:
                print(
                    f"    {source.display_name()}"
                )

            print(
                "Participant skipped to avoid using the wrong file."
            )

            return None

        # --------------------------------------------------------
        # No direct CSV.
        # Multiple ZIP sources remain ambiguous.
        # --------------------------------------------------------
        print(
            f"ERROR: {pid} has multiple valid EDA sources:"
        )

        for source in valid:
            print(
                f"    {source.display_name()}"
            )

        print(
            "Participant skipped to avoid using the wrong file."
        )

        return None

    # ------------------------------------------------------------
    # CASE 3: Candidates were discovered, but none contains
    # the required columns.
    # ------------------------------------------------------------
    print(
        f"ERROR: {pid} candidate file(s) found, "
        "but required columns are missing:"
    )

    for source in candidates:
        print(
            f"    {source.display_name()}"
        )

    return None

    # ------------------------------------------------------------
    # CASE 3: Candidates exist, but none passed validation
    # ------------------------------------------------------------
    print(
        f"ERROR: {pid} candidate file(s) found, "
        "but required columns are missing:"
    )

    for source in candidates:
        print(f"    {source.display_name()}")

    return None


# ============================================================
# 9. OUTPUT FILES
# ============================================================

def safe_condition(value: str) -> str:
    value = str(value or "Unknown").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value or "Unknown"


def output_path_for_window(
    participant_root: Path,
    pid: str,
    window: PhaseWindow,
) -> Path:

    # ========================================================
    # BASELINE
    # ========================================================
    if window.phase == "Baseline":

        return (
            participant_root
            / f"{pid}_Baseline_EDA.csv"
        )

    # ========================================================
    # 2-BACK
    #
    # Block 01-06 -> 2BackPart1
    # Block 07-12 -> 2BackPart2
    # ========================================================
    if window.phase == "2Back":

        if window.block is None:
            raise ValueError(
                f"{pid}: 2Back window has no block number."
            )

        if 1 <= window.block <= 6:
            part_folder = "2BackPart1"

        elif 7 <= window.block <= 12:
            part_folder = "2BackPart2"

        else:
            raise ValueError(
                f"{pid}: invalid 2Back block number: "
                f"{window.block}"
            )

        return (
            participant_root
            / part_folder
            / f"Block{window.block:02d}_EDA.csv"
        )

    # ========================================================
    # SPEECH / REST
    #
    # The filename depends on the actual condition:
    #
    # Control  -> P8_SpeechRest_Control_EDA.csv
    # Stressed -> P8_SpeechRest_Stressed_EDA.csv
    # ========================================================
    if window.phase == "SpeechRest":

        condition = safe_condition(window.condition)

        return (
            participant_root
            / f"{pid}_SpeechRest_{condition}_EDA.csv"
        )

    # ========================================================
    # SAFETY CHECK
    # ========================================================
    raise ValueError(
        f"{pid}: unknown phase: {window.phase}"
    )


# ============================================================
# 10. EXTRACT ONE PARTICIPANT
# ============================================================

def extract_participant(
    pid: str,
    source: EdaSource,
    windows: list[PhaseWindow],
):
    """
    Streams through the raw EDA file only ONCE.

    The Excel phase timestamps are the ground-truth phase windows.
    If the EDA recording covers only PART of a phase, DO NOT skip it.
    Extract every available EDA sample that lies inside that phase.

    Conceptually:
        covered_start = max(phase_start, EDA_recording_start)
        covered_end   = min(phase_end,   EDA_recording_end)

    Examples:
        Expected Baseline: 11:59:18 -> 12:01:18
        EDA starts at:     12:00:10
        Extract:           12:00:10 -> 12:01:18   (PARTIAL_START)

        Expected Block:    12:06:27 -> 12:07:27
        EDA ends at:       12:07:00
        Extract:           12:06:27 -> 12:07:00   (PARTIAL_END)

    If there is no EDA sample inside a phase, its phase CSV remains
    header-only and the summary reports NO_OVERLAP / NO_SAMPLES.

    No resampling.
    No averaging.
    No interpolation.
    No filtering.
    Negative/-1 sensor values are preserved exactly.
    """

    participant_root = OUTPUT_ROOT / pid
    participant_root.mkdir(parents=True, exist_ok=True)

    complete_marker = participant_root / "_COMPLETE.txt"

    if (
        SKIP_COMPLETED_PARTICIPANTS
        and complete_marker.exists()
    ):
        print(f"{pid}: already completed -> skipped")
        return "SKIPPED_COMPLETE", []

    expected_columns = required_columns_for_pid(pid)
    timestamp_column = f"{pid}_Timestamp_FormattedUnix_CAL"
    timestamp_output_index = expected_columns.index(timestamp_column)
    timestamp_decoder = EdaTimestampDecoder(windows)

    phase_files = {}
    phase_writers = {}
    sample_counts = {
        window.label: 0
        for window in windows
    }

    first_extracted_ts = {
        window.label: None
        for window in windows
    }
    last_extracted_ts = {
        window.label: None
        for window in windows
    }

    # First and last valid timestamps in the entire EDA recording.
    eda_first_ts = None
    eda_last_ts = None

    combined_file = None
    combined_writer = None

    try:
        # ----------------------------------------------------
        # Open output files first.
        # ----------------------------------------------------
        for window in windows:
            out_path = output_path_for_window(
                participant_root,
                pid,
                window,
            )
            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            f = out_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
            phase_files[window.label] = f

            writer = csv.writer(f)
            writer.writerow(expected_columns)
            phase_writers[window.label] = writer

        if WRITE_PARTICIPANT_COMBINED_CSV:
            combined_path = (
                participant_root
                / f"{pid}_All_Phases_EDA.csv"
            )

            combined_file = combined_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )

            combined_writer = csv.writer(combined_file)

            combined_writer.writerow(
                [
                    "ParticipantId",
                    "Phase",
                    "Block",
                    "SpeechOrRest",
                    "PhaseStart",
                    "PhaseEnd",
                ]
                + expected_columns
            )

        # ----------------------------------------------------
        # Stream through EDA source.
        # ----------------------------------------------------
        with open_eda_text(source) as text:
            header, delimiter = read_header_and_delimiter(text)

            missing = [
                col
                for col in expected_columns
                if (
                    col not in header
                    and col.removeprefix(f"{pid}_")
                    not in OPTIONAL_EDA_COLUMN_SUFFIXES
                )
            ]

            if missing:
                raise ValueError(
                    f"{pid}: required EDA columns missing:\n"
                    + "\n".join(missing)
                )

            optional_missing = [
                col
                for col in expected_columns
                if col not in header
            ]
            if optional_missing:
                print(
                    f"WARNING: {pid} optional source column(s) missing; "
                    f"blank output values will be written: "
                    + ", ".join(optional_missing)
                )

            column_indices = [
                header.index(col) if col in header else None
                for col in expected_columns
            ]

            timestamp_index = header.index(
                timestamp_column
            )

            reader = csv.reader(
                text,
                delimiter=delimiter,
            )

            # Windows are already chronological.
            windows_sorted = sorted(
                windows,
                key=lambda w: w.start,
            )

            current_idx = 0
            current = windows_sorted[current_idx]

            total_rows = 0
            valid_timestamp_rows = 0

            for row in reader:
                total_rows += 1

                # Trailing delimiter / short malformed row protection.
                if timestamp_index >= len(row):
                    continue

                ts = timestamp_decoder.parse(
                    row[timestamp_index]
                )

                # This automatically skips the units row:
                # yyyy/mm/dd hh:mm:ss.000
                if ts is None:
                    continue

                valid_timestamp_rows += 1

                if eda_first_ts is None:
                    eda_first_ts = ts
                eda_last_ts = ts

                # Advance through completed phase windows.
                while current_idx < len(windows_sorted):
                    current = windows_sorted[current_idx]

                    passed_end = (
                        ts > current.end
                        if INCLUDE_END_TIMESTAMP
                        else ts >= current.end
                    )

                    if passed_end:
                        current_idx += 1
                        continue

                    break

                # All experimental phases have finished.
                # Keep scanning so eda_last_ts represents the actual
                # end of the EDA recording.
                if current_idx >= len(windows_sorted):
                    continue

                current = windows_sorted[current_idx]

                if ts < current.start:
                    continue

                inside = (
                    current.start <= ts <= current.end
                    if INCLUDE_END_TIMESTAMP
                    else current.start <= ts < current.end
                )

                if not inside:
                    continue

                selected_values = [
                    row[i] if i is not None and i < len(row) else ""
                    for i in column_indices
                ]
                selected_values[timestamp_output_index] = (
                    format_output_timestamp(ts)
                )

                phase_writers[current.label].writerow(
                    selected_values
                )

                sample_counts[current.label] += 1

                if first_extracted_ts[current.label] is None:
                    first_extracted_ts[current.label] = ts
                last_extracted_ts[current.label] = ts

                if combined_writer:
                    combined_writer.writerow(
                        [
                            pid,
                            current.phase,
                            current.block or "",
                            current.condition,
                            current.start.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            current.end.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        ]
                        + selected_values
                    )

        # ----------------------------------------------------
        # Work out actual EDA coverage for every phase.
        # ----------------------------------------------------
        results = []

        for window in windows_sorted:
            expected_duration = (
                window.end - window.start
            ).total_seconds()

            overlap_start = None
            overlap_end = None
            overlap_duration = 0.0
            coverage_percent = 0.0

            count = sample_counts[window.label]
            first_sample = first_extracted_ts[window.label]
            last_sample = last_extracted_ts[window.label]

            if eda_first_ts is None or eda_last_ts is None:
                coverage_status = "NO_VALID_EDA_TIMESTAMPS"

            else:
                candidate_start = max(
                    window.start,
                    eda_first_ts,
                )
                candidate_end = min(
                    window.end,
                    eda_last_ts,
                )

                has_overlap = (
                    candidate_start <= candidate_end
                    if INCLUDE_END_TIMESTAMP
                    else candidate_start < candidate_end
                )

                if not has_overlap:
                    coverage_status = "NO_OVERLAP"

                else:
                    overlap_start = candidate_start
                    overlap_end = candidate_end
                    overlap_duration = max(
                        0.0,
                        (
                            overlap_end - overlap_start
                        ).total_seconds(),
                    )

                    if expected_duration > 0:
                        coverage_percent = min(
                            100.0,
                            max(
                                0.0,
                                (
                                    overlap_duration
                                    / expected_duration
                                )
                                * 100.0,
                            ),
                        )

                    missing_start = (
                        eda_first_ts > window.start
                    )
                    missing_end = (
                        eda_last_ts < window.end
                    )

                    if count == 0:
                        # The EDA file's overall time span intersects
                        # this phase, but no valid row was actually found
                        # inside it (for example, an internal data gap).
                        coverage_status = "OVERLAP_BUT_NO_SAMPLES"
                    elif missing_start and missing_end:
                        coverage_status = "PARTIAL_BOTH"
                    elif missing_start:
                        coverage_status = "PARTIAL_START"
                    elif missing_end:
                        coverage_status = "PARTIAL_END"
                    else:
                        coverage_status = "FULL"

            results.append(
                {
                    "phase": window.phase,
                    "block": window.block,
                    "condition": window.condition,
                    "start": window.start,
                    "end": window.end,
                    "expected_duration": expected_duration,
                    "eda_first": eda_first_ts,
                    "eda_last": eda_last_ts,
                    "covered_start": overlap_start,
                    "covered_end": overlap_end,
                    "covered_duration": overlap_duration,
                    "coverage_percent": coverage_percent,
                    "first_sample": first_sample,
                    "last_sample": last_sample,
                    "sample_count": count,
                    "coverage_status": coverage_status,
                    "timestamp_mode": timestamp_decoder.mode,
                    "timestamp_anchor": timestamp_decoder.anchor_text,
                }
            )

        # ----------------------------------------------------
        # Per-participant summary.
        # ----------------------------------------------------
        summary_path = (
            participant_root
            / f"{pid}_Phase_Summary.csv"
        )

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as summary_file:

            writer = csv.writer(summary_file)
            writer.writerow(
                [
                    "ParticipantId",
                    "Phase",
                    "Block",
                    "SpeechOrRest",
                    "ExpectedStart",
                    "ExpectedEnd",
                    "ExpectedDurationSeconds",
                    "EDARecordingStart",
                    "EDARecordingEnd",
                    "CoveredStart",
                    "CoveredEnd",
                    "CoveredDurationSeconds",
                    "CoveragePercent",
                    "FirstExtractedSample",
                    "LastExtractedSample",
                    "SampleCount",
                    "CoverageStatus",
                    "TimestampMode",
                    "TimestampAnchorHour",
                    "Source",
                ]
            )

            for result in results:
                writer.writerow(
                    [
                        pid,
                        result["phase"],
                        result["block"] or "",
                        result["condition"],
                        result["start"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        result["end"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        round(
                            result["expected_duration"],
                            3,
                        ),
                        (
                            result["eda_first"].strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )
                            if result["eda_first"]
                            else ""
                        ),
                        (
                            result["eda_last"].strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )
                            if result["eda_last"]
                            else ""
                        ),
                        (
                            result["covered_start"].strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )
                            if result["covered_start"]
                            else ""
                        ),
                        (
                            result["covered_end"].strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )
                            if result["covered_end"]
                            else ""
                        ),
                        round(
                            result["covered_duration"],
                            3,
                        ),
                        round(
                            result["coverage_percent"],
                            2,
                        ),
                        (
                            result["first_sample"].strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )
                            if result["first_sample"]
                            else ""
                        ),
                        (
                            result["last_sample"].strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )
                            if result["last_sample"]
                            else ""
                        ),
                        result["sample_count"],
                        result["coverage_status"],
                        result["timestamp_mode"],
                        result["timestamp_anchor"],
                        source.display_name(),
                    ]
                )

        complete_marker.write_text(
            (
                "Extraction completed successfully.\n"
                "Partial EDA phase coverage was preserved.\n"
            ),
            encoding="utf-8",
        )

        return "CREATED", results

    finally:
        for f in phase_files.values():
            try:
                f.close()
            except Exception:
                pass

        if combined_file:
            try:
                combined_file.close()
            except Exception:
                pass


# ============================================================
# 11. MAIN
# ============================================================

def main():
    participants = (
        PARTICIPANTS_TO_PROCESS
        if PARTICIPANTS_TO_PROCESS is not None
        else CORRECT_PARTICIPANTS
    )

    participants = [
        normalize_pid(pid)
        for pid in participants
    ]
    participants = [
        pid for pid in participants if pid
    ]
    participants = sorted(
        set(participants),
        key=pid_number,
    )

    print("=" * 78)
    print("ARYA EDA PHASE EXTRACTOR")
    print("=" * 78)
    print(f"Participants selected : {len(participants)}")
    print(f"EDA root              : {EDA_ROOT}")
    print(f"Master timing Excel   : {MASTER_XLSX}")
    print(f"Block timing Excel    : {BLOCKS_XLSX}")
    print(f"Output root           : {OUTPUT_ROOT}")
    print()
    print("Matching rule:")
    print("  EDA absolute timestamp vs experiment absolute timestamp")
    print("  Baseline   = BaselineStart -> BaselineEnd")
    print("  2-back     = P_Date + 2BackStart -> 2BackEnd, Block1...Block12")
    print("  SpeechRest = SR_Start -> SR_End")
    print("  If EDA covers only part of a phase, extract the available overlap.")
    print("  Partial phases are NOT skipped.")
    print()
    print("IMPORTANT:")
    print("  P_Vid_BS/P_Vid_BE, P_Vid_2Back_S/E and P_SR_S/E")
    print("  are VIDEO OFFSETS and are NOT used for EDA extraction.")
    print("=" * 78)
    print()

    master, block_tables = load_timing_tables()

    print("Scanning EDA ZIP/CSV files...")
    source_index = discover_eda_sources(EDA_ROOT)
    print(
        f"Participant IDs discovered in EDA filenames: "
        f"{len(source_index)}"
    )
    print()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_summary_path = (
        OUTPUT_ROOT
        / GLOBAL_SUMMARY_FILENAME
    )

    with global_summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as global_summary_file:

        global_writer = csv.writer(global_summary_file)
        global_writer.writerow(
            [
                "ParticipantId",
                "Status",
                "Phase",
                "Block",
                "StartTimestamp",
                "EndTimestamp",
                "SampleCount",
                "EDASource",
                "Message",
            ]
        )

        created = 0
        skipped = 0
        failed = 0

        for number, pid in enumerate(
            participants,
            start=1,
        ):
            print(
                f"[{number:03d}/{len(participants):03d}] "
                f"{pid}"
            )

            source = select_source_for_pid(
                pid,
                source_index,
            )

            if source is None:
                print("    -> NO VALID EDA SOURCE")
                global_writer.writerow(
                    [
                        pid,
                        "NO_EDA_SOURCE",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "No unique EDA source with required columns",
                    ]
                )
                failed += 1
                continue

            try:
                windows = build_phase_windows(
                    pid,
                    master,
                    block_tables,
                )

                print(
                    f"    source : {source.display_name()}"
                )

                for w in windows:
                    if w.block is not None:
                        label = f"Block{w.block:02d}"
                    else:
                        label = w.phase

                    print(
                        f"    {label:<12} "
                        f"{w.start} -> {w.end}"
                    )

                status, results = extract_participant(
                    pid,
                    source,
                    windows,
                )

                if status == "SKIPPED_COMPLETE":
                    skipped += 1
                    global_writer.writerow(
                        [
                            pid,
                            status,
                            "",
                            "",
                            "",
                            "",
                            "",
                            source.display_name(),
                            "",
                        ]
                    )
                    continue

                for result in results:
                    global_writer.writerow(
                        [
                            pid,
                            "CREATED",
                            result["phase"],
                            result["block"] or "",
                            result["start"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            result["end"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            result["sample_count"],
                            source.display_name(),
                            "",
                        ]
                    )

                created += 1

                print("    -> CREATED")
                for result in results:
                    label = (
                        f"Block{result['block']:02d}"
                        if result["block"] is not None
                        else result["phase"]
                    )
                    covered_start = (
                        result["covered_start"].strftime(
                            "%H:%M:%S.%f"
                        )[:-3]
                        if result["covered_start"]
                        else "-"
                    )
                    covered_end = (
                        result["covered_end"].strftime(
                            "%H:%M:%S.%f"
                        )[:-3]
                        if result["covered_end"]
                        else "-"
                    )

                    print(
                        f"       {label:<12}: "
                        f"{result['sample_count']:,} samples | "
                        f"{result['coverage_status']} | "
                        f"{result['coverage_percent']:.2f}% | "
                        f"{covered_start} -> {covered_end}"
                    )

            except Exception as exc:
                failed += 1

                print(f"    -> FAILED: {exc}")

                global_writer.writerow(
                    [
                        pid,
                        "FAILED",
                        "",
                        "",
                        "",
                        "",
                        "",
                        source.display_name(),
                        str(exc),
                    ]
                )

            global_summary_file.flush()
            print()

    print("=" * 78)
    print("FINISHED")
    print("=" * 78)
    print(f"Participants created : {created}")
    print(f"Participants skipped : {skipped}")
    print(f"Participants failed  : {failed}")
    print(f"Summary              : {global_summary_path}")
    print(f"Output               : {OUTPUT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        sys.exit(1)