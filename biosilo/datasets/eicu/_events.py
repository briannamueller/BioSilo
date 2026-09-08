"""Derive vasopressor and respiratory-support onset times from raw eICU tables.

* **Shock proxy**: first documented vasopressor infusion at a positive rate in
  ``infusionDrug``. The ``medication`` table is deliberately excluded because
  it records orders, which do not establish that a drug was administered.
* **ARF proxy**: the earliest of a non-zero mechanical-ventilation start offset,
  a positive PEEP limit, or PEEP/CPAP charting in ``respiratoryCharting``. This
  is an operational respiratory-support onset, not a diagnosis of ARF.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

CHUNK = 500_000

VASOPRESSORS = (
    "norepinephrine", "levophed",
    "epinephrine",
    "dopamine",
    "vasopressin",
    "phenylephrine", "neo-synephrine", "neosynephrine",
)

DEFINITIONS = {
    "shock": (
        "first positive-rate vasopressor infusion documented in infusionDrug"
    ),
    "arf": (
        "first non-zero ventilation-start offset, positive PEEP limit, or "
        "PEEP/CPAP charting; an operational respiratory-support proxy, not a "
        "diagnosis"
    ),
}


def definition(outcome: str) -> str:
    """Human-readable event definition stored with generated clients."""
    try:
        return DEFINITIONS[outcome]
    except KeyError as exc:
        raise ValueError(f"Unknown event outcome {outcome!r}") from exc


def find_table(eicu_dir: Path, table: str) -> Optional[Path]:
    """Locate a raw table case-insensitively, plain or gzipped.

    eICU distributions use inconsistent filename case, including
    ``infusiondrug.csv.gz`` versus ``infusionDrug``.
    """
    eicu_dir = Path(eicu_dir)
    wanted = {f"{table}.csv".lower(), f"{table}.csv.gz".lower()}

    for directory in (eicu_dir / "data", eicu_dir):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.name.lower() in wanted:
                return entry
    return None


def onset_times(eicu_dir: Path, outcome: str) -> Dict[int, float]:
    """First onset offset (minutes from unit admission) per patient."""
    if outcome == "shock":
        return _shock_onsets(eicu_dir)
    if outcome == "arf":
        return _arf_onsets(eicu_dir)
    raise ValueError(f"Unknown event outcome {outcome!r}")


def _is_vasopressor(name: str) -> bool:
    lowered = str(name).lower()
    return any(v in lowered for v in VASOPRESSORS)


def _shock_onsets(eicu_dir: Path) -> Dict[int, float]:
    rows = []

    infusion = find_table(eicu_dir, "infusionDrug")
    if infusion is not None:
        for chunk in pd.read_csv(
            infusion, chunksize=CHUNK,
            usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
        ):
            hit = chunk[
                chunk["drugname"].apply(_is_vasopressor)
                & (pd.to_numeric(chunk["drugrate"], errors="coerce") > 0)
            ]
            if len(hit):
                rows.append(hit[["patientunitstayid", "infusionoffset"]].rename(
                    columns={"patientunitstayid": "pid", "infusionoffset": "t"}))

    return _earliest(rows)


def _arf_onsets(eicu_dir: Path) -> Dict[int, float]:
    rows = []

    care = find_table(eicu_dir, "respiratoryCare")
    if care is not None:
        for chunk in pd.read_csv(
            care, chunksize=CHUNK,
            usecols=["patientunitstayid", "respcarestatusoffset",
                     "ventstartoffset", "peeplimit"],
        ):
            vent_offset = pd.to_numeric(
                chunk["ventstartoffset"], errors="coerce")
            # Zero commonly represents an unavailable date/time in this table.
            # Negative offsets are real pre-admission starts and must remain so
            # those stays are excluded from prospective event prediction.
            vent = chunk[vent_offset.notna() & vent_offset.ne(0)]
            if len(vent):
                rows.append(vent[["patientunitstayid", "ventstartoffset"]].rename(
                    columns={"patientunitstayid": "pid", "ventstartoffset": "t"}))

            peep = chunk[pd.to_numeric(chunk["peeplimit"], errors="coerce") > 0]
            if len(peep):
                rows.append(peep[["patientunitstayid", "respcarestatusoffset"]].rename(
                    columns={"patientunitstayid": "pid", "respcarestatusoffset": "t"}))

    charting = find_table(eicu_dir, "respiratoryCharting")
    if charting is not None:
        for chunk in pd.read_csv(
            charting, chunksize=CHUNK,
            usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel"],
        ):
            peep = chunk[chunk["respchartvaluelabel"].astype(str)
                         .str.contains("PEEP|CPAP", case=False, na=False)]
            if len(peep):
                rows.append(peep[["patientunitstayid", "respchartoffset"]].rename(
                    columns={"patientunitstayid": "pid", "respchartoffset": "t"}))

    return _earliest(rows)


def _earliest(rows) -> Dict[int, float]:
    if not rows:
        return {}
    allrows = pd.concat(rows, ignore_index=True)
    allrows["t"] = pd.to_numeric(allrows["t"], errors="coerce")
    allrows = allrows.dropna(subset=["t"])
    return allrows.groupby("pid")["t"].min().to_dict()
