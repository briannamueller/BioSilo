# TCGA

Bulk tumour transcriptomics from The Cancer Genome Atlas, federated by the
hospital that contributed each sample. 33 cancer cohorts available.

## Getting the data

Open access: no credentials or data-use agreement. BioSilo provides two source
backends:

**Xena** (default). Pre-normalized log2(TPM+1) matrices from UCSC Xena and the
only supported source for survival endpoints.

**GDC.** STAR counts from the GDC API. BioSilo normalizes them by default;
`expression_transform="none"` retains raw counts for custom workflows.

GDC STAR-count files identify aliquots, whereas its clinical endpoint uses case
IDs. BioSilo preserves each full aliquot barcode, stores GDC's explicit
aliquot-to-case relationship, and joins clinical fields through it. A case may
supply multiple aliquots, which correctly receive the same case-level clinical
values; an aliquot mapped to multiple cases is rejected. Data prepared by older
sample-level handling are detected and refreshed instead of being reused.

Download and prepare the selected source before generating a partition:

```python
import biosilo

biosilo.download(
    "TCGA",
    source="xena",
    cohorts=("LUAD", "LUSC", "BRCA", "COAD"),
)
```

Use `source="gdc"` to prepare GDC data instead. The helper stores prepared data
under `data/_raw/TCGA/<source>/` by default and reuses complete files on later
runs. Pass `root=` to change the common data root or `source_dir=` to change
only the TCGA source directory.

Acquisition is separate from partition generation: `biosilo.generate(...)`
never downloads TCGA data. If the requested source is absent or incomplete, it
reports the missing files and directs you to run `biosilo.download(...)`.
Downloads are written to temporary sibling files, validated, and atomically
installed so an interruption cannot leave a partial file that appears complete.
The pan-cancer Xena expression matrix is about 700 MB; GDC data are larger.
BioSilo does not redistribute either source.

```python
import biosilo

path = biosilo.generate(
    "TCGA", source="xena",
)
data = biosilo.load("TCGA", partition=path.name)
```

## Task

Three, chosen per run:

| task | type | meaning |
|---|---|---|
| `cancer_type` | multi-class | which cancer this sample is (needs ≥2 cohorts) |
| `stage` | binary | advanced (stage IV) versus not |
| `survival` | binary | survived past a threshold versus not |

`cancer_type` requires at least two cohorts. `survival` requires Xena because
the GDC clinical download used here has no survival endpoints.

## Sample shape

`(n_genes,)` float32. Genes expressed in fewer than 10% of samples are dropped,
then the top `top_genes` by variance are kept, 2,000 by default.

Xena data arrives already log2(TPM+1) normalized. With the default
`expression_transform="source_default"`, GDC raw counts are converted per
sample to CPM and then `log2(x+1)`; Xena is left unchanged. Set
`expression_transform="none"` to retain raw GDC counts.

The feature-selection protocol is inductive. Patient-level hospital splits are
established first, then the nonzero filter and variance ranking are fitted on
the combined training samples and applied unchanged to test samples.

## Federation

By **contributing hospital**, which is TCGA's real collection structure:
different institutions contributed different cancer types, patient populations,
and sample counts.

A TCGA barcode encodes its tissue source site (`TCGA-A7-A0CG-01` uses site
`A7`), and the bundled `centers.csv` maps sites to institutions. The mapping is
many-to-one, so multiple site codes may be merged into one hospital client.

Hospitals contributing fewer than `min_samples` are dropped.

### Site-code matching

BioSilo removes leading zeroes from both barcode site codes and `centers.csv`
codes before lookup. Thus barcode site `05` matches table site `5` and joins the
same hospital client as its other mapped site codes.

## Groups: patient

A TCGA barcode identifies a **sample**, not a person. One patient can contribute
several (a primary tumour and a metastasis, say), and those share germline
genotype and much of their expression profile, so a split between them leaks.

The patient id is the first three barcode fields: `TCGA-A7-A0CG-01A-11R-A00Z-07`
→ `TCGA-A7-A0CG`. Where a patient contributed once, the group-aware split is
identical to a plain stratified one.

## Partitioning

BioSilo exposes only the hospital partition described above. There is no
partition-mode parameter. Synthetic Dirichlet or pathological repartitioning
can be applied by downstream frameworks. A by-cohort partition is not provided
because it would group `cancer_type` samples by their target label.

## Parameters

| | default | |
|---|---|---|
| `cohorts` | `("LUAD", "LUSC", "BRCA", "COAD")` | any of the 33; ≥2 for `cancer_type` |
| `task` | `cancer_type` | see above |
| `source` | `xena` | `xena` or `gdc` |
| `expression_transform` | `source_default` | GDC: CPM + log2; Xena: unchanged; `none` disables it |
| `top_genes` | 2000 | 0 keeps every surviving gene |
| `min_samples` | 20 | minimum per hospital |
| `min_nonzero_fraction` | 0.10 | drop near-silent genes |
| `survival_threshold_days` | 1095 | 3 years, `survival` only |
| `train_ratio` | 0.80 | stratified, group-aware |
| `seed` | 1 | controls the per-hospital split |
| `source_dir` | `<root>/_raw/TCGA` | prepared Xena or GDC source; location only |

## The 33 cohorts

```
ACC BLCA BRCA CESC CHOL COAD DLBC ESCA GBM HNSC KICH KIRC KIRP LAML LGG
LIHC LUAD LUSC MESO OV PAAD PCPG PRAD READ SARC SKCM STAD TGCT THCA THYM
UCEC UCS UVM
```

## Storage

`npz` per client: `x`, `y`, and group ids.

## Citations

Cite TCGA itself, and the Xena platform if you use that backend.

> Goldman, M. J., Craft, B., Hastie, M., Repečka, K., McDade, F., Kamath, A., et
> al. (2020). Visualizing and interpreting cancer genomics data via the Xena
> platform. *Nature Biotechnology*, 38, 675–678.

TCGA's own publication guidelines state how the consortium expects to be
credited, and per-cohort marker papers exist for many of the 33.
