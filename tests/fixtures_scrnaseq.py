"""Create small scRNA-seq matrices with the Zenodo archive layout.

Synthetic expression values exercise cross-study gene intersection, label
harmonization, and Muraro chromosome suffix handling. Optional source
``Labels.csv`` files provide coverage against the published vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from biosilo.datasets.scrnaseq import STUDIES

#: Synthetic vocabulary covering every configured synonym.
SYNTHETIC_LABELS = {
    "BaronHuman":  ["alpha", "beta", "delta", "gamma", "ductal",
                    "activated_stellate", "quiescent_stellate", "schwann"],
    "Muraro":      ["alpha", "beta", "delta", "pp", "duct", "mesenchymal"],
    "Segerstolpe": ["alpha", "beta", "delta", "gamma", "PSC", "co-expression"],
    "Xin":         ["alpha", "beta", "delta", "gamma"],
}

#: Genes each study measures. The intersection is what a client ends up with;
#: Muraro's carry chromosome suffixes, as the real file does.
GENES = {
    "BaronHuman":  ["INS", "GCG", "SST", "PPY", "KRT19", "BARONONLY"],
    "Muraro":      ["INS__chr11", "GCG__chr2", "SST__chr3", "PPY__chr17",
                    "KRT19__chr17", "MURAROONLY__chr1"],
    "Segerstolpe": ["INS", "GCG", "SST", "PPY", "KRT19", "SEGONLY"],
    "Xin":         ["INS", "GCG", "SST", "PPY", "KRT19", "XINONLY"],
}

#: Expected gene intersection.
SHARED_GENES = ["GCG", "INS", "KRT19", "PPY", "SST"]

#: Expected cell-type intersection after harmonization.
SHARED_TYPES = {"alpha", "beta", "delta", "gamma"}

PANCREAS = ("BaronHuman", "Muraro", "Segerstolpe", "Xin")


def build(root: Path, labels_dir: Optional[Path] = None, per_type: int = 12) -> Path:
    """Write a fake archive. Returns the directory to pass as ``source_dir``.

    With ``labels_dir``, real ``Labels.csv`` files are copied in and expression
    is generated to match their row counts.
    """
    raw = root / "raw"
    rng = np.random.default_rng(0)

    for study in PANCREAS:
        subdir, csv_name = STUDIES[study]
        directory = raw / "Intra-dataset" / subdir
        directory.mkdir(parents=True, exist_ok=True)

        if labels_dir is not None:
            labels = pd.read_csv(labels_dir / f"{study}.csv", header=0).iloc[:, 0]
        else:
            labels = pd.Series(
                [t for t in SYNTHETIC_LABELS[study] for _ in range(per_type)])

        n = len(labels)
        cells = [f"{study}_cell{i}" for i in range(n)]
        genes = GENES[study]

        # Counts, not normalized values; preprocessing does CPM itself.
        x = pd.DataFrame(
            rng.integers(0, 500, size=(n, len(genes))).astype(float),
            index=cells, columns=genes,
        )
        x.to_csv(directory / csv_name)
        pd.DataFrame({"x": labels.to_numpy()}).to_csv(
            directory / "Labels.csv", index=False)

    return raw
