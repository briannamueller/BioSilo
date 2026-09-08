"""Create a small eICU-compatible processed feature store.

The fixture covers downstream cohort and split behavior with:

* a hospital below the size floor
* a hospital with only one label present
* two stays belonging to one person (a readmission)
* stays too short to fill the observation window

Stage-one preprocessing has separate optional tests against the eICU demo and a
Fed-eICU reference store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_TS_FEATURES = 3      # values; masks double this
N_FLAT = 4
N_DIAG = 3

#: (hospital_id, n_stays, labels) for the cohort the filters must sort out.
COHORT = [
    (1, 20, "mixed"),    # qualifies, and holds the readmission
    (2, 12, "mixed"),    # qualifies
    (3, 3, "mixed"),     # too small
    (4, 15, "single"),   # one class only
]

SHORT_STAY_HOURS = 10    # below a 24h window, so these must be dropped


def build(root: Path, hours: int = 30) -> tuple[Path, Path]:
    """Write a fake processed store and patient table. Returns (processed, eicu)."""
    processed = root / "processed"
    eicu = root / "eicu"
    (processed / "train").mkdir(parents=True, exist_ok=True)
    eicu.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows, ts_rows, flat_rows, diag_rows, label_rows = [], [], [], [], []

    pid = 1000
    person = 0
    for hospital, n, kind in COHORT:
        for k in range(n):
            pid += 1
            person += 1

            # One readmission: the second stay in hospital 1 reuses a person.
            uniquepid = f"P{person:04d}"
            if hospital == 1 and k == 1:
                uniquepid = f"P{person - 1:04d}"
            if hospital == 2 and k == 0:
                # The same person later appears at another hospital. A split
                # performed independently inside each hospital could leak it.
                uniquepid = "P0001"

            # Two stays in hospital 2 are too short to fill the window.
            n_hours = SHORT_STAY_HOURS if (hospital == 2 and k < 2) else hours

            y = 0 if kind == "single" else int(k % 2)

            rows.append((pid, hospital, uniquepid))
            label_rows.append({
                "patient": pid,
                "actualhospitalmortality": "EXPIRED" if y else "ALIVE",
                "unitdischargeoffset": float(n_hours * 60 + 600),
            })
            flat_rows.append({
                "patient": pid,
                **{f"flat_{i}": float(rng.random()) for i in range(N_FLAT)},
                # a hospital-level column, which drop_hospital_vars must remove
                "region_midwest": float(hospital == 1),
            })
            diag_rows.append({
                "patient": pid,
                **{f"diag_{i}": float(rng.integers(0, 2)) for i in range(N_DIAG)},
            })
            for h in range(1, n_hours + 1):
                ts_rows.append({
                    "patient": pid,
                    "time": h,
                    "hour": (h % 24) / 23.0,
                    **{f"v_{i}": float(rng.random()) for i in range(N_TS_FEATURES)},
                    **{f"v_{i}_mask": 1.0 for i in range(N_TS_FEATURES)},
                })

    split = processed / "train"
    (split / "stays.txt").write_text("\n".join(str(r[0]) for r in rows) + "\n")
    pd.DataFrame(label_rows).to_csv(split / "labels.csv", index=False)
    pd.DataFrame(flat_rows).to_csv(split / "flat.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(split / "diagnoses.csv", index=False)
    pd.DataFrame(ts_rows).to_csv(split / "timeseries.csv", index=False)

    pd.DataFrame(
        [{"patientunitstayid": p, "hospitalid": h, "uniquepid": u} for p, h, u in rows]
    ).to_csv(eicu / "patient.csv", index=False)

    return processed, eicu
