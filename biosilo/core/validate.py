"""Generation-time shape, length, and group checks."""

from __future__ import annotations

import numpy as np

from .contract import ClientData, Split, input_fields, n_samples


class ValidationError(Exception):
    """Raised when generated client data violates the shared contract."""


def check_split(split: Split, *, where: str) -> None:
    """First axis is the sample axis for every field, and lengths agree."""
    X, y, groups = split
    fields = input_fields(X)

    n = n_samples(X)
    for name, arr in fields:
        if len(arr) != n:
            raise ValidationError(
                f"{where}: input '{name}' has {len(arr)} rows but the sample "
                f"axis is {n}. Every field's first axis must be the sample axis."
            )
    if len(y) != n:
        raise ValidationError(f"{where}: {n} samples but {len(y)} labels.")
    if groups is not None and len(groups) != n:
        raise ValidationError(f"{where}: {n} samples but {len(groups)} group ids.")


def check_groups_disjoint(client: ClientData) -> None:
    """Require train and test to contain disjoint group IDs."""
    _, _, gtrain = client.train
    _, _, gtest = client.test

    if gtrain is None and gtest is None:
        return
    if (gtrain is None) != (gtest is None):
        raise ValidationError(
            f"client {client.client_id}: one split has group ids and the other "
            "does not. Either both carry them or neither does."
        )

    straddling = np.intersect1d(np.asarray(gtrain), np.asarray(gtest))
    if straddling.size:
        shown = ", ".join(str(g) for g in straddling[:5])
        more = "" if straddling.size <= 5 else f" (+{straddling.size - 5} more)"
        raise ValidationError(
            f"client {client.client_id}: {straddling.size} group(s) appear in "
            f"both train and test: {shown}{more}. Split by whole groups, or "
            "validation and test end up holding near-duplicates."
        )


def check_consistent(spec, first_spec, grouped: bool, first_grouped: bool,
                     *, client_id: str) -> None:
    """Require one input specification and grouping mode across all clients."""
    if spec != first_spec:
        raise ValidationError(
            f"client {client_id} has inputs {spec}, but earlier clients have "
            f"{first_spec}. Every client in a partition must share one shape."
        )
    if grouped != first_grouped:
        have, lacks = ("has", "do not") if grouped else ("does not have", "do")
        raise ValidationError(
            f"client {client_id} {have} group ids while earlier clients {lacks}. "
            "Every client in a partition must use the same grouping mode."
        )


def check_client(client: ClientData) -> None:
    check_split(client.train, where=f"client {client.client_id} train")
    check_split(client.test, where=f"client {client.client_id} test")
    check_groups_disjoint(client)
