#!/usr/bin/env python3
"""Aggregate-only audit of BioSilo's eICU shock and ARF event definitions.

The report contains counts and value categories, never patient identifiers or
raw clinical rows. It is intended to run beside a licensed eICU installation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

CHUNK_SIZE = 500_000
VASOPRESSORS = (
    "norepinephrine", "levophed", "epinephrine", "dopamine", "vasopressin",
    "phenylephrine", "neo-synephrine", "neosynephrine",
)


def find_table(root: Path, name: str) -> Path | None:
    wanted = {f"{name}.csv".lower(), f"{name}.csv.gz".lower()}
    for directory in (root / "data", root):
        if directory.is_dir():
            for entry in directory.iterdir():
                if entry.is_file() and entry.name.lower() in wanted:
                    return entry
    return None


def drug_match(values):
    lowered = values.astype(str).str.casefold()
    match = pd.Series(False, index=values.index)
    for drug in VASOPRESSORS:
        match |= lowered.str.contains(drug, regex=False, na=False)
    return match


def update_earliest(target: dict[int, float], patients, offsets) -> None:
    frame = pd.DataFrame({"patient": patients, "offset": offsets})
    frame["offset"] = pd.to_numeric(frame["offset"], errors="coerce")
    frame = frame.dropna(subset=["patient", "offset"])
    if frame.empty:
        return
    for patient, offset in frame.groupby("patient")["offset"].min().items():
        patient = int(patient)
        value = float(offset)
        target[patient] = min(value, target.get(patient, value))


def merge_earliest(*sources) -> dict[int, float]:
    merged: dict[int, float] = {}
    for source in sources:
        for patient, value in source.items():
            merged[patient] = min(value, merged.get(patient, value))
    return merged


def onset_summary(onsets: dict[int, float]) -> dict:
    values = pd.Series(list(onsets.values()), dtype=float)
    return {
        "patients": int(len(values)),
        "before_or_at_admission": int((values <= 0).sum()),
        "through_4h": int((values <= 240).sum()),
        "exactly_4h": int((values == 240).sum()),
        "through_12h": int((values <= 720).sum()),
        "exactly_12h": int((values == 720).sum()),
    }


def audit_shock(root: Path) -> dict:
    infusion_current: dict[int, float] = {}
    medication_current: dict[int, float] = {}
    medication_uncancelled: dict[int, float] = {}
    infusion_rows = Counter()
    medication_rows = Counter()
    cancellation_values = Counter()

    infusion = find_table(root, "infusionDrug")
    if infusion:
        for chunk in pd.read_csv(
            infusion, chunksize=CHUNK_SIZE, low_memory=False,
            usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
        ):
            hit = chunk.loc[drug_match(chunk["drugname"])].copy()
            rate = pd.to_numeric(hit["drugrate"], errors="coerce")
            infusion_rows["matching_name"] += len(hit)
            infusion_rows["positive_rate"] += int((rate > 0).sum())
            infusion_rows["zero_rate"] += int((rate == 0).sum())
            infusion_rows["negative_rate"] += int((rate < 0).sum())
            infusion_rows["missing_or_text_rate"] += int(rate.isna().sum())
            positive = hit.loc[rate > 0]
            update_earliest(
                infusion_current, positive["patientunitstayid"],
                positive["infusionoffset"])

    medication = find_table(root, "medication")
    if medication:
        for chunk in pd.read_csv(
            medication, chunksize=CHUNK_SIZE, low_memory=False,
            usecols=["patientunitstayid", "drugstartoffset", "drugordercancelled",
                     "drugname", "dosage"],
        ):
            hit = chunk.loc[drug_match(chunk["drugname"])].copy()
            starts = pd.to_numeric(hit["drugstartoffset"], errors="coerce")
            cancelled = hit["drugordercancelled"].fillna("<missing>").astype(str)
            cancellation_values.update(cancelled.tolist())
            medication_rows["matching_name"] += len(hit)
            medication_rows["numeric_start"] += int(starts.notna().sum())
            medication_rows["missing_start"] += int(starts.isna().sum())
            dosage = pd.to_numeric(hit["dosage"], errors="coerce")
            medication_rows["positive_numeric_dosage"] += int((dosage > 0).sum())
            medication_rows["non_numeric_or_missing_dosage"] += int(dosage.isna().sum())
            valid = hit.loc[starts.notna()]
            update_earliest(
                medication_current, valid["patientunitstayid"],
                valid["drugstartoffset"])
            not_cancelled = cancelled.str.casefold().isin(
                {"no", "false", "0", "n", "<missing>", ""})
            valid = hit.loc[starts.notna() & not_cancelled]
            update_earliest(
                medication_uncancelled, valid["patientunitstayid"],
                valid["drugstartoffset"])

    current = merge_earliest(infusion_current, medication_current)
    uncancelled = merge_earliest(infusion_current, medication_uncancelled)
    return {
        "infusion_rows": dict(infusion_rows),
        "medication_rows": dict(medication_rows),
        "medication_cancellation_values": dict(cancellation_values),
        "infusion_only": onset_summary(infusion_current),
        "medication_orders_only": onset_summary(medication_current),
        "combined_infusion_and_medication_orders": onset_summary(current),
        "excluding_cancelled_medication_orders": onset_summary(uncancelled),
    }


def audit_arf(root: Path) -> dict:
    vent_positive: dict[int, float] = {}
    vent_negative: dict[int, float] = {}
    vent_nonzero: dict[int, float] = {}
    peep_numeric: dict[int, float] = {}
    peep_positive: dict[int, float] = {}
    chart_any: dict[int, float] = {}
    chart_peep: dict[int, float] = {}
    care_rows = Counter()
    chart_rows = Counter()
    chart_labels = Counter()

    care = find_table(root, "respiratoryCare")
    if care:
        for chunk in pd.read_csv(
            care, chunksize=CHUNK_SIZE, low_memory=False,
            usecols=["patientunitstayid", "respcarestatusoffset",
                     "ventstartoffset", "peeplimit"],
        ):
            vent = pd.to_numeric(chunk["ventstartoffset"], errors="coerce")
            peep = pd.to_numeric(chunk["peeplimit"], errors="coerce")
            care_rows["vent_positive"] += int((vent > 0).sum())
            care_rows["vent_zero"] += int((vent == 0).sum())
            care_rows["vent_negative"] += int((vent < 0).sum())
            care_rows["peep_positive"] += int((peep > 0).sum())
            care_rows["peep_zero"] += int((peep == 0).sum())
            care_rows["peep_negative"] += int((peep < 0).sum())
            update_earliest(
                vent_positive, chunk.loc[vent > 0, "patientunitstayid"],
                vent.loc[vent > 0])
            update_earliest(
                vent_negative, chunk.loc[vent < 0, "patientunitstayid"],
                vent.loc[vent < 0])
            update_earliest(
                vent_nonzero, chunk.loc[vent.ne(0) & vent.notna(),
                                        "patientunitstayid"],
                vent.loc[vent.ne(0) & vent.notna()])
            update_earliest(
                peep_numeric, chunk.loc[peep.notna(), "patientunitstayid"],
                chunk.loc[peep.notna(), "respcarestatusoffset"])
            update_earliest(
                peep_positive, chunk.loc[peep > 0, "patientunitstayid"],
                chunk.loc[peep > 0, "respcarestatusoffset"])

    charting = find_table(root, "respiratoryCharting")
    if charting:
        for chunk in pd.read_csv(
            charting, chunksize=CHUNK_SIZE, low_memory=False,
            usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel"],
        ):
            labels = chunk["respchartvaluelabel"].astype(str)
            any_match = labels.str.contains("PEEP|CPAP", case=False, na=False)
            peep_match = labels.str.contains("PEEP", case=False, na=False)
            chart_rows["peep_or_cpap"] += int(any_match.sum())
            chart_rows["peep"] += int(peep_match.sum())
            chart_rows["cpap_without_peep"] += int((any_match & ~peep_match).sum())
            chart_labels.update(labels[any_match].tolist())
            update_earliest(
                chart_any, chunk.loc[any_match, "patientunitstayid"],
                chunk.loc[any_match, "respchartoffset"])
            update_earliest(
                chart_peep, chunk.loc[peep_match, "patientunitstayid"],
                chunk.loc[peep_match, "respchartoffset"])

    current = merge_earliest(vent_positive, peep_numeric, chart_any)
    positive_peep_current_vent = merge_earliest(
        vent_positive, peep_positive, chart_any)
    positive_peep_peep_labels = merge_earliest(
        vent_positive, peep_positive, chart_peep)
    production = merge_earliest(vent_nonzero, peep_positive, chart_any)
    production_peep_labels = merge_earliest(
        vent_nonzero, peep_positive, chart_peep)
    return {
        "respiratory_care_rows": dict(care_rows),
        "respiratory_charting_rows": dict(chart_rows),
        "matching_chart_labels": dict(chart_labels.most_common()),
        "vent_start_positive_only": onset_summary(vent_positive),
        "vent_start_negative_only": onset_summary(vent_negative),
        "vent_start_nonzero_only": onset_summary(vent_nonzero),
        "positive_peep_limit_only": onset_summary(peep_positive),
        "chart_peep_or_cpap_only": onset_summary(chart_any),
        "numeric_peep_limit_definition": onset_summary(current),
        "require_positive_peep_limit": onset_summary(positive_peep_current_vent),
        "also_exclude_cpap_only_labels_keep_current_vent": onset_summary(
            positive_peep_peep_labels),
        "production_definition": onset_summary(production),
        "production_without_cpap_only_labels": onset_summary(
            production_peep_labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "privacy": "aggregate counts only; no patient identifiers or raw rows",
        "shock": audit_shock(args.source_dir),
        "arf": audit_arf(args.source_dir),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
