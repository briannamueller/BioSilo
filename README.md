# BioSilo

Naturally partitioned biomedical datasets for federated learning.

BioSilo preserves client boundaries already present in source data, such as
hospitals, imaging institutions, and sequencing studies.

## Install

Install BioSilo from PyPI:

```bash
pip install biosilo
```

BioSilo requires Python 3.9 or newer. The base installation depends only on NumPy. Install the optional dependencies for the datasets you intend to use:

```bash
pip install "biosilo[eicu]"
pip install "biosilo[scrnaseq]"
pip install "biosilo[brats]"
pip install "biosilo[tcga]"
pip install "biosilo[all]"       # every dataset extra
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/briannamueller/BioSilo.git
cd BioSilo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

## Quickstart

`biosilo.generate(...)` creates or reuses a partition on disk. `biosilo.load(...)` opens a completed partition.
`biosilo.expected_partition(...)` returns the path determined by the same
generation parameters without creating it.

The built-in synthetic dataset provides a quick installation check:

```python
import biosilo

path = biosilo.generate("synthetic")

p = biosilo.load("synthetic", partition=path.name)
p.num_clients        # 3
p.client_ids         # ['site-0', 'site-1', 'site-2']

X, y, groups = p.client(0, "train")
X.shape, y.shape     # ((18, 5), (18,))
```

### Dataset setup and storage

Each supported dataset has a guide in [`docs/datasets/`](https://github.com/briannamueller/BioSilo/tree/main/docs/datasets) that
explains how to obtain and prepare its required data. Complete those steps
before generating a partition.

BioSilo uses `data/` in the current working directory as its default root:

```text
data/
├── _raw/
│   └── <Dataset>/            # source data used for generation
├── _cache/
│   └── <Dataset>/            # reusable preprocessing results
└── <Dataset>/
    └── <partition_id>/       # completed generated partition
```

When applicable, `source_dir` defaults to `<root>/_raw/<Dataset>/` and
`cache_dir` defaults to `<root>/_cache/<Dataset>/`. Completed partitions are
always written to `<root>/<Dataset>/<partition_id>/`.

For example, with the eICU source data saved under `data/_raw/eICU/`, the following generates a partition using the default paths:

```python
path = biosilo.generate(
    "eICU",
    task="mortality_24h",             # docs/datasets/eicu.md lists all eight
    num_clients=10,
)
p = biosilo.load("eICU", partition=path.name)
```
The reusable output from the expensive eICU preprocessing pipeline is saved under `data/_cache/eICU/` and the completed partition is saved under `data/eICU/<partition_id>/`.

To override the default locations, pass `root`, `source_dir`, or `cache_dir`
when generating a partition. The root can also be set globally with
`BIOSILO_DATA_ROOT`.

By default, `generate` reuses a completed partition for the same configuration.
New partitions are built and validated in a temporary directory before being
moved to their final location, so a failed build does not leave a partial
partition behind. Pass `overwrite=True` only when you want to regenerate and
replace an existing partition.

`biosilo.available()` lists the datasets supported by generation and loading.
Each dataset's guide states whether BioSilo also provides a download helper.

## The data contract

Some datasets contain multiple samples from the same subject, forming a group. Examples include multiple brain-image slices from one subject and multiple ICU stays from one patient. During generation, BioSilo ensures that no group ID appears in both a client’s training and test data to prevent subject-level data leakage.

| Return | Meaning |
|---|---|
| `X` | Input data: a NumPy array or, when each sample has multiple input components, a named tuple of arrays. |
| `y` | int array, one label per sample. |
| `groups` | One group ID per sample, or `None` when the samples are independent. |

Load one client’s training or test data with:

```python
X, y, groups = p.client(0, "train")
```

BioSilo only creates training and test splits, leaving users to define validation splits according to their needs. For datasets that provide group IDs, pass `groups` to a group-aware splitter such as scikit-learn’s `GroupShuffleSplit` so that all samples from one subject remain together.

Most datasets return a single input array, so `X[i]` contains the complete input for sample `i`.
Some datasets return multiple input arrays per sample. In eICU, for example, each ICU stay has time-series measurements and static patient features:

```python
time_series = X.ts[i]
static_features = X.static[i]
```

Datasets stored as memmapped `.npy` files (currently BraTS) are not loaded into
memory in their entirety. `np.memmap` subclasses `ndarray`, so this does not
change how you index them. Other datasets use compressed per-client `.npz`
files.

## Datasets

| Dataset | Task | Clients | Sample | Groups |
|---|---|---|---|---|
| [**eICU**](https://github.com/briannamueller/BioSilo/blob/main/docs/datasets/eicu.md) | binary: mortality, length of stay, vasopressor infusion, or respiratory-support onset | hospital | `ts (T, n_ts)` + `static (n_static,)` | person |
| [**scRNAseq**](https://github.com/briannamueller/BioSilo/blob/main/docs/datasets/scrnaseq.md) | cell-type annotation, 12 classes | sequencing study | `(n_genes,)` | none |
| [**BraTS**](https://github.com/briannamueller/BioSilo/blob/main/docs/datasets/brats.md) | binary: tumour present per axial slice | imaging institution | `(4, 240, 240)` | subject |
| [**TCGA**](https://github.com/briannamueller/BioSilo/blob/main/docs/datasets/tcga.md) | cancer type, stage, or survival | contributing hospital | `(n_genes,)` | patient |

**No dataset is redistributed here.** Each dataset has a dedicated guide in
[`docs/datasets/`](https://github.com/briannamueller/BioSilo/tree/main/docs/datasets) covering data acquisition, preprocessing,
parameters, and licensing.

## Contributing

The dataset-module contract, cache-versioning policy, and contributor test
workflow are in [`CONTRIBUTING.md`](https://github.com/briannamueller/BioSilo/blob/main/CONTRIBUTING.md).

## Citing BioSilo

[`CITATION.cff`](https://github.com/briannamueller/BioSilo/blob/main/CITATION.cff) carries the machine-readable citation; GitHub
renders it under **Cite this repository**. The datasets have their own
citations, listed on each page in [`docs/datasets/`](https://github.com/briannamueller/BioSilo/tree/main/docs/datasets), and those
are the ones the data providers require.

## Licence

MIT, covering the source code only. See [LICENSE](https://github.com/briannamueller/BioSilo/blob/main/LICENSE).

The datasets are not MIT. Each carries its own terms, and BioSilo redistributes
none of them. See the per-dataset pages in [`docs/datasets/`](https://github.com/briannamueller/BioSilo/tree/main/docs/datasets).
