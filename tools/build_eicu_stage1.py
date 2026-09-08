#!/usr/bin/env python3
"""Build one parameter-keyed eICU feature store for a task horizon."""
from __future__ import annotations

import argparse
from pathlib import Path

from biosilo.datasets.eicu._stage1 import Stage1Params, ensure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int, required=True)
    parser.add_argument(
        "--diagnosis-hours", type=int,
        help="Diagnosis window; defaults to min(5, --hours)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    diagnosis_hours = (
        args.diagnosis_hours
        if args.diagnosis_hours is not None
        else min(5, args.hours)
    )
    if not 1 <= diagnosis_hours <= min(5, args.hours):
        parser.error("--diagnosis-hours must be between 1 and min(5, --hours)")

    store, provenance = ensure(
        args.cache_dir,
        args.source_dir,
        Stage1Params(
            observation_hours=args.hours,
            diagnosis_hours=diagnosis_hours,
            train_ratio=args.train_ratio,
            split_seed=args.seed,
        ),
    )
    print(f"store={store}")
    print(f"provenance={provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
