# BraTS

Brain tumour detection from multi-parametric MRI, federated by the institution
that contributed each scan. From the FeTS 2022 challenge data, which ships an
explicit institution mapping.

## Getting the data

Requires a [Synapse](https://www.synapse.org/) account and **accepted data-use
terms** for the FeTS 2022 dataset, `syn28546456`. Approval is not automatic.

```bash
pip install synapseclient nibabel
synapse login          # or set SYNAPSE_AUTH_TOKEN
```

Several gigabytes of NIfTI volumes. Extract the training data under
`data/_raw/BraTS/`: this directory should contain the `FeTS2022_*` subject
folders and `partitioning_1.csv`. Pass `source_dir` if it is stored elsewhere.

BioSilo does not redistribute these files.

```python
import biosilo

path = biosilo.generate(
    "BraTS",
)
data = biosilo.load("BraTS", partition=path.name)
```

## Task

**Binary, per 2D axial slice: does this slice contain tumour?**

The FeTS segmentation mask is collapsed to `any(seg > 0)`: 1 if the slice
contains any tumour voxel of any subtype, 0 otherwise.

This is slice-level classification, not segmentation, grading, or subtype
classification.

## Sample shape

`(4, H, W)` float32: four MRI modalities as channels, in this fixed order:

```
T1 · T1Gd · T2 · T2-FLAIR
```

Nothing resizes, so H×W is the native FeTS in-plane size, **240×240**.

Each modality is normalized per volume: clip to its 0.5/99.5 percentiles over
brain voxels, then z-score using statistics from the complete 3D volume.
Background remains zero for the brain-fraction filter.

Slices with under 1% non-zero voxels are dropped as effectively empty.

## Federation

By institution, from `partitioning_1.csv`. **14 institutions qualify** at the
default thresholds, with 1,194 subjects and 79,546 slices at stride 2.

The client-size distribution is highly uneven:

```
institution  1 → 511 subjects · 34,068 slices
institution 18 → 382 subjects · 25,649 slices
the other 12   →  11–47 subjects each
```

Two institutions hold roughly **75% of the data**. Report both per-client and
sample-weighted metrics so this imbalance is visible.

## Groups: subject

Adjacent axial slices share anatomy, acquisition, and intensity distribution.
Splits therefore hold out whole subjects. Generation rejects any client whose
subject IDs overlap between train and test.

## Parameters

| | default | |
|---|---|---|
| `source` | `FeTS2022` | the challenge release the partitioning CSV comes from |
| `slice_stride` | 2 | take every Nth axial slice |
| `min_brain_fraction` | 0.01 | drop near-empty slices |
| `min_subjects_per_institution` | 5 | drop small institutions |
| `min_slices_per_institution` | 100 | as above |
| `num_clients` | 0 | maximum clients; 0 keeps every qualifying institution; a shortfall is reported |
| `train_ratio` | 0.80 | split by subject |
| `seed` | 1 | controls the per-institution subject split |
| `source_dir` | `<root>/_raw/BraTS` | extracted FeTS data; location only |

## Storage

**Memmapped `.npy`**, one file per client per split. Input arrays are loaded
lazily from disk.

## Generation streams

The largest institution is 34,068 slices, or about **31 GB** materialized.

A first pass reads **T1 only** to get exact slice counts (the brain-fraction
filter only looks at the first channel), so the output array can be allocated on
disk up front and filled subject by subject. Peak stays under a
gigabyte regardless of institution size.

## Subjects with inconsistent geometry

Subjects must have internally consistent modality shapes and match the cohort's
dominant in-plane geometry. Nonconforming subjects are skipped. FeTS itself is
uniformly 240×240.

## Citations

Cite the BraTS benchmark, the segmentation labels, and the FeTS challenge.

> Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby,
> J., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark
> (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024.

> Bakas, S., Akbari, H., Sotiras, A., Bilello, M., Rozycki, M., Kirby, J. S., et
> al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with
> expert segmentation labels and radiomic features. *Scientific Data*, 4, 170117.

> Pati, S., Baid, U., Edwards, B., et al. (2022). Federated learning enables big
> data for rare cancer boundary detection. *Nature Communications*, 13, 7346.

The Synapse landing page for `syn28546456` states the challenge's own required
citations authoritatively.
