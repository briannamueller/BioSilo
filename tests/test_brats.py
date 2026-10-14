"""BraTS filtering, grouping, shape, and streaming tests."""

from __future__ import annotations

import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

import biosilo
import fixtures_brats as fx
from biosilo.core.contract import hashable_params
from biosilo.datasets import brats


def _generate(root, **overrides):
    data = fx.build(root)
    kwargs = dict(
        slice_stride=1, min_brain_fraction=0.01,
        min_subjects_per_institution=5, min_slices_per_institution=10,
        train_ratio=0.75, seed=1, source_dir=str(data),
    )
    kwargs.update(overrides)
    out = biosilo.generate("BraTS", root=root / "data", **kwargs)
    return biosilo.load("BraTS", root=root / "data", partition=out.name)


# Federation

def test_institutions_below_the_floor_are_dropped(root):
    p = _generate(root)
    # institution 9 has 2 subjects, under the floor of 5
    assert "9" not in p.client_ids, p.client_ids
    assert "1" in p.client_ids, p.client_ids


def test_requested_client_shortfall_is_reported(root):
    output = StringIO()
    with redirect_stdout(output):
        p = _generate(root, num_clients=8)
    noun = "client" if p.num_clients == 1 else "clients"
    assert f"Only {p.num_clients} {noun} survived the filters; requested 8" \
        in output.getvalue()


def test_unusable_subjects_are_skipped(root):
    """Institution 2 has 5 subjects but one lacks FLAIR and one has a bad shape."""
    p = _generate(root, min_subjects_per_institution=3)
    idx = p.client_ids.index("2")
    metadata = p.manifest["clients"][idx]["metadata"]
    assert metadata["n_subjects"] == 3, metadata


# Slices

def test_blank_slices_are_filtered_out(root):
    p = _generate(root)
    idx = p.client_ids.index("1")
    client = p.manifest["clients"][idx]
    total = sum(client[split] for split in ("train", "test"))
    # Six subjects times the retained brain slices.
    assert total == 6 * len(fx.BRAIN_SLICES), total
    assert total < 6 * fx.DEPTH


def test_samples_are_four_channel(root):
    p = _generate(root)
    assert p.inputs[0]["shape"] == [4, fx.H, fx.W], p.inputs[0]
    assert p.num_classes == 2, p.num_classes
    X, y, groups = p.client(0, "train")
    assert X.shape[1:] == (4, fx.H, fx.W), X.shape
    assert set(np.unique(y)) <= {0, 1}, np.unique(y)


def test_stride_halves_the_slice_count(root):
    dense = _generate(root)
    with tempfile.TemporaryDirectory() as other:
        sparse = _generate(Path(other), slice_stride=2)
    a = sum(client[split] for client in dense.manifest["clients"]
            for split in ("train", "test"))
    b = sum(client[split] for client in sparse.manifest["clients"]
            for split in ("train", "test"))
    assert b < a, (a, b)


# Groups

def test_subjects_never_straddle_a_split(root):
    p = _generate(root)
    assert p.has_groups and p.group_unit == "subject", p.group_unit
    for cid in range(p.num_clients):
        _, _, gtr = p.client(cid, "train")
        _, _, gte = p.client(cid, "test")
        assert np.intersect1d(gtr, gte).size == 0, f"client {cid} leaks a subject"


def test_group_ids_are_subject_names(root):
    p = _generate(root)
    _, _, groups = p.client(0, "train")
    assert all(str(g).startswith("FeTS2022_") for g in groups[:5]), groups[:5]
    # every slice of a subject is contiguous and shares one id
    assert len(np.unique(groups)) < len(groups)


# Streaming

def test_partitions_are_memmapped_not_materialized(root):
    """BraTS inputs must use the memmap backend."""
    p = _generate(root)
    assert p.manifest["storage"] == "memmap", p.manifest["storage"]
    X, _, _ = p.client(0, "train")
    assert isinstance(X, np.memmap), type(X)


def test_build_yields_memmaps_and_cleans_up(root):
    """The builder removes scratch memmaps after iteration."""
    data = fx.build(root)
    params = brats.Params(
        slice_stride=1, min_subjects_per_institution=5,
        min_slices_per_institution=10, source_dir=str(data))

    seen = []
    for client in brats.build(params):
        X, y, groups = client.train
        assert isinstance(X, np.memmap), type(X)
        assert len(X) == len(y) == len(groups)
        seen.append(Path(X.filename))

    # the generator's finally clause removes its workspace
    assert not any(path.exists() for path in seen), seen


def test_partition_label_uses_seed_and_client_count():
    assert brats.label(brats.Params(seed=3, slice_stride=1)) == "s3"
    assert brats.label(brats.Params(seed=3, slice_stride=1, num_clients=8)) == "s3_n8"


def test_source_is_fixed():
    params = brats.Params()
    assert params.source == "FeTS2022"
    assert hashable_params(params)["source"] == "FeTS2022"
    with pytest.raises(TypeError):
        brats.Params(source="another-release")
