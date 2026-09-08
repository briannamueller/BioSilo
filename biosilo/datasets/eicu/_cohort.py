"""Select eICU hospital clients and split their stays."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SORT_MODES = ("size", "positives", "prevalence")


def qualifying(
    hospital_ids: np.ndarray,
    summary: np.ndarray,
    min_size: int,
    min_minority: int,
    min_prev: float,
) -> List[Tuple[int, np.ndarray]]:
    """Return hospitals meeting size, class-count, and prevalence thresholds."""
    kept = []
    floor = max(min_size, 2)

    for hid in np.unique(hospital_ids):
        idx = np.flatnonzero(hospital_ids == hid)
        if len(idx) < floor:
            continue

        classes, counts = np.unique(summary[idx], return_counts=True)
        if len(classes) < 2:
            continue
        if min_minority > 0 and int(counts.min()) < min_minority:
            continue
        if min_prev > 0 and _prevalence(classes, counts) < min_prev:
            continue

        kept.append((int(hid), idx))

    return kept


def _prevalence(classes: np.ndarray, counts: np.ndarray) -> float:
    return float(_positives(classes, counts) / counts.sum())


def _positives(classes: np.ndarray, counts: np.ndarray) -> int:
    """Count labels greater than zero."""
    return int(sum(c for cls, c in zip(classes, counts) if cls > 0))


def rank(
    candidates: List[Tuple[int, np.ndarray]],
    summary: np.ndarray,
    sort_mode: str,
    num_clients: int,
) -> List[Tuple[int, np.ndarray]]:
    """Order hospitals by the chosen criterion and keep the top ``num_clients``."""
    if sort_mode not in SORT_MODES:
        raise ValueError(f"sort_mode must be one of {SORT_MODES}, got {sort_mode!r}")

    def score(idx: np.ndarray):
        classes, counts = np.unique(summary[idx], return_counts=True)
        if sort_mode == "size":
            return len(idx)
        if sort_mode == "positives":
            return _positives(classes, counts)
        return _prevalence(classes, counts)

    ordered = sorted(candidates, key=lambda item: score(item[1]), reverse=True)
    if num_clients > 0:
        ordered = ordered[:num_clients]
    return ordered


def split(
    summary: np.ndarray,
    groups: Optional[np.ndarray],
    train_ratio: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split one hospital, preserving labels and repeated-person groups."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    n = len(summary)
    classes, counts = np.unique(summary, return_counts=True)
    stratify = summary if counts.min() >= 2 else None

    repeated = groups is not None and len(np.unique(groups)) < n
    if not repeated:
        train_idx, test_idx = train_test_split(
            np.arange(n),
            train_size=train_ratio,
            stratify=stratify,
            random_state=random_state,
        )
        return np.sort(train_idx), np.sort(test_idx)

    # Select the fold count nearest the requested test ratio.
    n_splits = max(2, min(int(round(1.0 / max(1e-9, 1.0 - train_ratio))), len(np.unique(groups))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(n), summary, groups))
    return np.sort(train_idx), np.sort(test_idx)


def global_person_split(
    summary: np.ndarray,
    persons: np.ndarray,
    train_ratio: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign every person once so they cannot cross partitions via hospitals."""
    from sklearn.model_selection import train_test_split

    unique_persons, inverse = np.unique(persons, return_inverse=True)
    person_labels = np.zeros(len(unique_persons), dtype=summary.dtype)
    np.maximum.at(person_labels, inverse, summary)
    classes, counts = np.unique(person_labels, return_counts=True)
    stratify = person_labels if len(classes) > 1 and counts.min() >= 2 else None
    train_persons, test_persons = train_test_split(
        unique_persons,
        train_size=train_ratio,
        stratify=stratify,
        random_state=random_state,
    )
    train = np.isin(persons, train_persons)
    test = np.isin(persons, test_persons)
    if np.any(train & test) or not np.all(train | test):
        raise AssertionError('global eICU person split is not a partition')
    return train, test


def label_counts(summary: np.ndarray) -> Dict[str, int]:
    classes, counts = np.unique(summary, return_counts=True)
    return {str(int(c)): int(n) for c, n in zip(classes, counts)}
