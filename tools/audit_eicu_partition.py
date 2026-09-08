#!/usr/bin/env python3
"""Generate and validate one complete eICU task partition.

The report contains aggregate counts only. It never prints patient or person
identifiers and is suitable for logs beside a licensed eICU installation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import biosilo
from biosilo.datasets.eicu import EVENT_TASKS, SCHEMA_VERSION, TASKS


def _inputs(x):
    return list(x) if hasattr(x, "_fields") else [x]


def _validate_partition(partition, task: str) -> dict:
    expected_hours = TASKS[task]
    if partition.schema_version != SCHEMA_VERSION:
        raise AssertionError(
            f"schema {partition.schema_version}, expected {SCHEMA_VERSION}")
    if partition.settings["task"] != task:
        raise AssertionError(f"partition task is {partition.settings['task']!r}")
    if partition.num_classes != 2:
        raise AssertionError(f"expected binary labels, got {partition.num_classes}")
    if not partition.has_groups or "person" not in (partition.group_unit or ""):
        raise AssertionError("partition does not expose person-level groups")

    train_people, test_people = set(), set()
    sample_counts = Counter()
    label_counts = Counter()

    for client_index in range(partition.num_clients):
        meta = partition.manifest["clients"][client_index]["metadata"]
        if meta["max_seq_len"] != expected_hours:
            raise AssertionError("client metadata has the wrong observation window")
        expected_diagnosis_hours = min(5, expected_hours)
        if meta["diagnosis_hours"] != expected_diagnosis_hours:
            raise AssertionError("client metadata has the wrong diagnosis window")
        if meta["preprocessing_protocol"] != "inductive":
            raise AssertionError("full audit requires an inductive BioSilo store")
        if task in EVENT_TASKS and not meta.get("event_definition"):
            raise AssertionError("event task is missing its operational definition")

        for split in ("train", "test"):
            x, y, groups = partition.client(client_index, split)
            arrays = _inputs(x)
            if len(arrays) != 2:
                raise AssertionError(f"expected two inputs, found {len(arrays)}")
            ts, static = map(np.asarray, arrays)
            y = np.asarray(y)
            groups = np.asarray(groups)

            n = len(y)
            if not (len(ts) == len(static) == len(groups) == n):
                raise AssertionError("input, label, and group lengths differ")
            if ts.ndim != 3 or ts.shape[1] != expected_hours:
                raise AssertionError(f"unexpected time-series shape {ts.shape}")
            if static.ndim != 2:
                raise AssertionError(f"unexpected static shape {static.shape}")
            if not np.isfinite(ts).all() or not np.isfinite(static).all():
                raise AssertionError("partition contains non-finite input values")
            unique_labels = set(np.unique(y).tolist())
            if not unique_labels.issubset({0, 1}):
                raise AssertionError(f"non-binary labels found: {unique_labels}")

            people = set(groups.tolist())
            if split == "train":
                train_people.update(people)
            else:
                test_people.update(people)
            sample_counts[split] += n
            label_counts.update(int(value) for value in y.tolist())

    overlap = train_people.intersection(test_people)
    if overlap:
        raise AssertionError(
            f"{len(overlap)} people occur in both train and test partitions")
    if set(label_counts) != {0, 1}:
        raise AssertionError(f"partition does not contain both labels: {label_counts}")

    return {
        "task": task,
        "schema_version": partition.schema_version,
        "clients": partition.num_clients,
        "samples": dict(sample_counts),
        "labels": {str(key): value for key, value in sorted(label_counts.items())},
        "unique_people": {
            "train": len(train_people),
            "test": len(test_people),
        },
        "train_test_person_overlap": 0,
        "input_shapes": partition.inputs,
        "checks": "passed",
        "privacy": "aggregate counts only; no patient or person identifiers",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    path = biosilo.generate(
        "eICU",
        root=args.root,
        overwrite=args.overwrite,
        task=args.task,
        source_dir=str(args.source_dir),
        cache_dir=str(args.cache_dir),
    )
    partition = biosilo.load("eICU", root=args.root, partition=path.name)
    report = _validate_partition(partition, args.task)
    report["partition"] = path.name
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
