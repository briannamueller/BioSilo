"""Core generation, validation, storage, identity, and loading tests."""

from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import biosilo
from biosilo.core import generate as generate_core
from biosilo.core import manifest, partition_id, validate
from biosilo.core.contract import ClientData
from biosilo.datasets import synthetic

PASSED, FAILED = [], []


def case(fn):
    """Run one check and record its outcome."""
    def run(root):
        try:
            fn(root)
        except Exception as exc:  # noqa: BLE001 - report all custom-harness failures
            FAILED.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            PASSED.append(fn.__name__)
    return run


# ── round trip ────────────────────────────────────────────────────────────────

@case
def single_input_round_trip(root):
    d = biosilo.generate("synthetic", root=root, n_clients=3, n_per_client=20)
    p = biosilo.load("synthetic", root=root, partition=d.name)

    assert (d / "manifest.json").is_file()
    assert not (d / "config.json").exists()
    assert p.manifest["settings"]["n_clients"] == 3
    assert p.manifest["input_spec"][0]["name"] == "x"
    assert p.manifest["target_spec"]["num_classes"] == p.num_classes
    assert [client["client_id"] for client in p.manifest["clients"]] == [
        "site-0", "site-1", "site-2"
    ]
    assert all(
        len(client["train_label_hist"]) == p.num_classes
        and len(client["test_label_hist"]) == p.num_classes
        for client in p.manifest["clients"]
    )

    assert p.num_clients == 3, p.num_clients
    assert p.client_ids == ["site-0", "site-1", "site-2"], p.client_ids
    assert p.is_multi_input is False
    assert p.has_groups is False
    assert len(p.inputs) == 1 and p.inputs[0]["name"] == "x"

    X, y, groups = p.client(1, "train")
    assert isinstance(X, np.ndarray), type(X)
    assert groups is None
    assert len(X) == len(y)
    # Train and test together contain the complete client.
    Xte, yte, _ = p.client(1, "test")
    assert len(y) + len(yte) == 20


@case
def multi_input_round_trip(root):
    d = biosilo.generate("synthetic", root=root, n_inputs=2, n_clients=2, seed=1)
    p = biosilo.load("synthetic", root=root, partition=d.name)

    assert p.is_multi_input is True
    names = [i["name"] for i in p.inputs]
    assert names == ["seq", "flat"], names
    # shapes exclude the sample axis
    assert p.inputs[0]["shape"] == [8, 5], p.inputs[0]
    assert p.inputs[1]["shape"] == [4], p.inputs[1]

    X, y, _ = p.client(0, "train")
    assert hasattr(X, "_fields") and X._fields == ("seq", "flat")
    # first axis is the sample axis for every field
    assert len(X.seq) == len(X.flat) == len(y)
    # per-sample indexing by declared order, the way a consumer does it
    sample = tuple(getattr(X, n)[0] for n in names)
    assert sample[0].shape == (8, 5) and sample[1].shape == (4,)


@case
def groups_survive_round_trip(root):
    d = biosilo.generate("synthetic", root=root, with_groups=True, n_per_client=24, seed=2)
    p = biosilo.load("synthetic", root=root, partition=d.name)

    assert p.has_groups is True
    assert p.group_unit == "block", p.group_unit

    _, _, gtr = p.client(0, "train")
    _, _, gte = p.client(0, "test")
    assert gtr is not None and gte is not None
    overlap = np.intersect1d(gtr, gte)
    assert overlap.size == 0, f"groups straddle the split: {overlap}"


# ── storage backends ──────────────────────────────────────────────────────────

@case
def memmap_backend_is_lazy_and_equivalent(root):
    args = dict(root=root, n_clients=2, n_per_client=16, seed=3)
    a = biosilo.generate("synthetic", **args)
    b = biosilo.generate("synthetic_memmap", **args)

    Xa, ya, _ = biosilo.load("synthetic", root=root, partition=a.name).client(0, "train")
    Xb, yb, _ = biosilo.load("synthetic_memmap", root=root, partition=b.name).client(0, "train")

    # the memmap partition is actually memory-mapped ...
    assert isinstance(Xb, np.memmap), type(Xb)
    # ... and a memmap is an ndarray, so a consumer cannot tell the difference
    assert isinstance(Xb, np.ndarray)
    np.testing.assert_array_equal(np.asarray(Xa), np.asarray(Xb))
    np.testing.assert_array_equal(ya, yb)


# ── validation ────────────────────────────────────────────────────────────────

@case
def straddling_group_is_refused(root):
    """Overlapping group IDs must fail validation."""
    x = np.zeros((6, 2), dtype=np.float32)
    y = np.zeros(6, dtype=np.int64)
    bad = ClientData(
        client_id="leaky",
        train=(x, y, np.array([1, 1, 2, 2, 3, 3])),
        test=(x, y, np.array([3, 3, 4, 4, 5, 5])),   # group 3 on both sides
    )
    try:
        validate.check_client(bad)
    except validate.ValidationError as exc:
        assert "both train and test" in str(exc), str(exc)
    else:
        raise AssertionError("a straddling group was accepted")


@case
def mismatched_lengths_are_refused(root):
    bad = ClientData(
        client_id="ragged",
        train=(np.zeros((5, 2), np.float32), np.zeros(4, np.int64), None),
        test=(np.zeros((2, 2), np.float32), np.zeros(2, np.int64), None),
    )
    try:
        validate.check_client(bad)
    except validate.ValidationError as exc:
        assert "labels" in str(exc), str(exc)
    else:
        raise AssertionError("mismatched X/y lengths were accepted")


@case
def overwrite_clears_stale_clients(root):
    """Rebuilding with fewer clients removes prior client files."""
    first = biosilo.generate("synthetic", root=root, n_clients=5, seed=1)

    original = synthetic.build
    synthetic.build = lambda p: (c for i, c in enumerate(original(p)) if i < 2)
    try:
        biosilo.generate("synthetic", root=root, n_clients=5, seed=1, overwrite=True)
    finally:
        synthetic.build = original

    on_disk = sorted((first / "train").glob("*.npz"))
    p = biosilo.load("synthetic", root=root, partition=first.name)
    assert len(on_disk) == p.num_clients == 2, (len(on_disk), p.num_clients)


@case
def failed_overwrite_leaves_the_existing_partition_intact(root):
    """A builder failure may not damage a previously complete partition."""
    part = biosilo.generate("synthetic", root=root, n_clients=3, seed=31)
    before_manifest = (part / manifest.MANIFEST_NAME).read_bytes()
    before_files = sorted(p.relative_to(part) for p in part.rglob("*"))

    original = synthetic.build

    def fail_after_one(params):
        yield next(iter(original(params)))
        raise RuntimeError("injected overwrite failure")

    synthetic.build = fail_after_one
    try:
        try:
            biosilo.generate(
                "synthetic", root=root, n_clients=3, seed=31, overwrite=True)
        except RuntimeError as exc:
            assert "injected" in str(exc)
        else:
            raise AssertionError("injected overwrite failure did not propagate")
    finally:
        synthetic.build = original

    assert (part / manifest.MANIFEST_NAME).read_bytes() == before_manifest
    assert sorted(p.relative_to(part) for p in part.rglob("*")) == before_files
    assert not list(part.parent.glob(f".{part.name}.building-*"))


@case
def failed_install_restores_the_existing_partition(root):
    """Failure after moving the old directory restores it from the backup."""
    final = root / "complete"
    staged = root / "staged"
    final.mkdir()
    staged.mkdir()
    (final / "identity.txt").write_text("old")
    (staged / "identity.txt").write_text("new")

    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected rename failure")
        return __import__("os").replace(source, destination)

    try:
        generate_core._install_staged_partition(staged, final, replace=fail_second)
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("injected install failure did not propagate")

    assert (final / "identity.txt").read_text() == "old"


@case
def clients_must_agree_on_shape(root):
    """Every client must match the partition input specification."""
    def mixed(_):
        yield ClientData("a", (np.zeros((4, 3), np.float32), np.zeros(4, np.int64), None),
                              (np.zeros((2, 3), np.float32), np.zeros(2, np.int64), None))
        yield ClientData("b", (np.zeros((4, 9), np.float32), np.zeros(4, np.int64), None),
                              (np.zeros((2, 9), np.float32), np.zeros(2, np.int64), None))

    original = synthetic.build
    synthetic.build = mixed
    try:
        biosilo.generate("synthetic", root=root, seed=42)
    except validate.ValidationError as exc:
        assert "must share one shape" in str(exc), str(exc)
    else:
        raise AssertionError("mismatched client shapes were accepted")
    finally:
        synthetic.build = original


@case
def clients_must_agree_on_groups(root):
    """Every client must use the same grouping mode."""
    def half(_):
        yield ClientData("a", (np.zeros((4, 3), np.float32), np.zeros(4, np.int64), np.array([0, 0, 1, 1])),
                              (np.zeros((2, 3), np.float32), np.zeros(2, np.int64), np.array([2, 2])))
        yield ClientData("b", (np.zeros((4, 3), np.float32), np.zeros(4, np.int64), None),
                              (np.zeros((2, 3), np.float32), np.zeros(2, np.int64), None))

    original = synthetic.build
    synthetic.build = half
    try:
        biosilo.generate("synthetic", root=root, seed=43)
    except validate.ValidationError as exc:
        assert "group ids" in str(exc), str(exc)
    else:
        raise AssertionError("inconsistent group presence was accepted")
    finally:
        synthetic.build = original


# ── identity ──────────────────────────────────────────────────────────────────

@case
def same_params_same_directory(root):
    a = biosilo.generate("synthetic", root=root, n_clients=2, seed=7)
    b = biosilo.generate("synthetic", root=root, n_clients=2, seed=7)
    assert a == b, (a, b)


@case
def expected_partition_resolves_without_generating(root):
    expected = biosilo.expected_partition(
        "synthetic", root=root, n_clients=2, seed=71
    )

    assert not expected.exists()
    assert biosilo.generate(
        "synthetic", root=root, n_clients=2, seed=71
    ) == expected


@case
def different_params_different_directory(root):
    a = biosilo.generate("synthetic", root=root, n_clients=2, seed=8)
    b = biosilo.generate("synthetic", root=root, n_clients=2, seed=9)
    assert a != b, (a, b)


@case
def partition_id_is_verifiable_and_shell_safe(root):
    d = biosilo.generate("synthetic", root=root, n_clients=2, seed=10)
    value = manifest.read(d)
    assert partition_id.verify(
        value["partition_id"], "synthetic", value["settings"],
        value["schema_version"])
    assert not (set(d.name) - set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )), f"unsafe characters in {d.name!r}"


@case
def paths_do_not_affect_identity(root):
    """Fields marked unhashed do not change partition identity."""
    from dataclasses import dataclass, field as dc_field

    from biosilo.core.contract import hashable_params, unhashed

    @dataclass(frozen=True)
    class P:
        seed: int = 1
        source_dir: str = unhashed("")

    assert hashable_params(P(seed=1, source_dir="/on/laptop")) == \
           hashable_params(P(seed=1, source_dir="/on/hpc"))


@case
def schema_version_changes_identity_but_provenance_does_not(root):
    original_schema = synthetic.SCHEMA_VERSION
    try:
        a = biosilo.generate(
            "synthetic", root=root, n_clients=2, seed=70, version="1.0.0")
        same = biosilo.generate(
            "synthetic", root=root, n_clients=2, seed=70, version="99.0.0")
        assert same == a, (a, same)

        synthetic.SCHEMA_VERSION = original_schema + 1
        changed = biosilo.generate(
            "synthetic", root=root, n_clients=2, seed=70, version="1.0.0")
        assert changed != a, (a, changed)
        assert manifest.read(changed)["schema_version"] == original_schema + 1
    finally:
        synthetic.SCHEMA_VERSION = original_schema


@case
def synthetic_labels_use_client_count_and_seed(root):
    from biosilo.datasets import synthetic_memmap

    params = synthetic.Params(n_clients=4, n_classes=9, n_inputs=2,
                              with_groups=True, seed=7)
    assert synthetic.label(params) == "n4_s7"
    assert synthetic_memmap.label(params) == "n4_s7"


# ── loader behaviour ──────────────────────────────────────────────────────────

@case
def default_root_is_data_in_the_current_directory(root):
    previous_cwd = Path.cwd()
    previous_env = os.environ.pop("BIOSILO_DATA_ROOT", None)
    try:
        os.chdir(root)
        path = biosilo.generate("synthetic", n_clients=2, seed=12)
        assert path.parent.parent.resolve() == (root / "data").resolve(), path
        loaded = biosilo.load("synthetic", partition=path.name)
        assert loaded.path == path
    finally:
        os.chdir(previous_cwd)
        if previous_env is not None:
            os.environ["BIOSILO_DATA_ROOT"] = previous_env


@case
def dataset_paths_follow_the_resolved_root(root):
    from dataclasses import dataclass

    from biosilo.core.contract import unhashed

    @dataclass(frozen=True)
    class SourceOnly:
        source_dir: str = unhashed("")

    @dataclass(frozen=True)
    class SourceAndCache:
        source_dir: str = unhashed("")
        cache_dir: str = unhashed("")

    data = root / "chosen-data"
    resolved = generate_core._resolve_dataset_paths(
        "BraTS", SourceOnly(), data)
    assert Path(resolved.source_dir) == data / "_raw" / "BraTS"

    resolved = generate_core._resolve_dataset_paths(
        "eICU", SourceAndCache(), data)
    assert Path(resolved.source_dir) == data / "_raw" / "eICU"
    assert Path(resolved.cache_dir) == data / "_cache" / "eICU"

    explicit = SourceAndCache(source_dir="/source", cache_dir="/cache")
    resolved = generate_core._resolve_dataset_paths("eICU", explicit, data)
    assert resolved.source_dir == "/source"
    assert resolved.cache_dir == "/cache"

@case
def val_split_is_rejected(root):
    biosilo.generate("synthetic", root=root, n_clients=2, seed=11)
    p = biosilo.load("synthetic", root=root, partition=_sole(root, "synthetic", seed=11))
    try:
        p.client(0, "val")
    except ValueError as exc:
        assert "carve it from train" in str(exc), str(exc)
    else:
        raise AssertionError("a 'val' split was accepted")


@case
def ambiguous_partition_is_refused(root):
    biosilo.generate("synthetic", root=root, n_clients=2, seed=20)
    biosilo.generate("synthetic", root=root, n_clients=2, seed=21)
    try:
        biosilo.load("synthetic", root=root)
    except ValueError as exc:
        assert "pass partition=" in str(exc), str(exc)
    else:
        raise AssertionError("an ambiguous partition was resolved without error")


@case
def unknown_dataset_is_refused(root):
    try:
        biosilo.generate("not-a-dataset", root=root)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown dataset accepted")


@case
def download_without_a_dataset_helper_is_refused(root):
    try:
        biosilo.download("synthetic", root=root)
    except NotImplementedError as exc:
        assert "guide" in str(exc)
    else:
        raise AssertionError("a dataset without a download helper was accepted")


def _sole(root, dataset, **kw):
    """Directory name of the partition generated with these params."""
    return biosilo.generate(dataset, root=root, **kw).name


# ── run ───────────────────────────────────────────────────────────────────────

CASES = [
    single_input_round_trip,
    multi_input_round_trip,
    groups_survive_round_trip,
    memmap_backend_is_lazy_and_equivalent,
    straddling_group_is_refused,
    overwrite_clears_stale_clients,
    failed_overwrite_leaves_the_existing_partition_intact,
    failed_install_restores_the_existing_partition,
    clients_must_agree_on_shape,
    clients_must_agree_on_groups,
    mismatched_lengths_are_refused,
    same_params_same_directory,
    expected_partition_resolves_without_generating,
    different_params_different_directory,
    partition_id_is_verifiable_and_shell_safe,
    paths_do_not_affect_identity,
    schema_version_changes_identity_but_provenance_does_not,
    synthetic_labels_use_client_count_and_seed,
    default_root_is_data_in_the_current_directory,
    dataset_paths_follow_the_resolved_root,
    val_split_is_rejected,
    unknown_dataset_is_refused,
    download_without_a_dataset_helper_is_refused,
]


def main() -> int:
    for fn in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))

    # This case requires two partitions under one root.
    with tempfile.TemporaryDirectory() as tmp:
        ambiguous_partition_is_refused(Path(tmp))

    for name in PASSED:
        print(f"  ok    {name}")
    for line in FAILED:
        print(f"  FAIL  {line}")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
