# scRNAseq

Cell-type annotation from single-cell RNA sequencing, federated by sequencing
study. Four independent human pancreas studies, one per client.

## Getting the data

Open access, no credentials. Zenodo record **3357167**, the cell-identification
benchmark collection of Abdelaal et al. (2019), which bundles several single-cell
datasets in a common format.

```bash
pip install zenodo-get
zenodo_get 10.5281/zenodo.3357167 -o data/_raw/scRNAseq
```

Extract it so that `data/_raw/scRNAseq/` contains an `Intra-dataset/`
directory. Pass `source_dir` if the extracted data is stored elsewhere.

```python
import biosilo

path = biosilo.generate(
    "scRNAseq",
)
data = biosilo.load("scRNAseq", partition=path.name)
```

Roughly 12 GB extracted. The pancreas studies alone are ~790 MB of CSV:

```
Baron (human)  301 MB   8,569 cells
Muraro         239 MB   2,122 cells
Segerstolpe    107 MB   2,133 cells
Xin            143 MB   1,449 cells
```

## Task

Multi-class classification of each cell's type. Every cell is one sample; the
features are its gene expression profile.

The default cohort is `pancreas`, pooling the four studies above. **12 classes**
with default settings:

```
alpha 4,865 · beta 3,702 · ductal 1,703 · acinar 1,359 · delta 950
gamma 633 · stellate 591 · endothelial 288 · macrophage 55 · mast 30
epsilon 28 · schwann 13
```

Severely imbalanced: alpha outnumbers schwann by a factor of 374.

## Sample shape

`(n_genes,)` float32. Genes are the **intersection across all studies in the
cohort**: 15,582 for pancreas, from studies carrying 17,499 to 33,889 each.

The ~11% of Baron's genes that don't survive are mostly gene-symbol version
drift. Segerstolpe uses an older HGNC release (`RFWD2`, `PRUNE`, `C1orf21`), so
biologically identical genes fail to match on name. All canonical pancreas
markers survive: INS, GCG, SST, PPY, KRT19, PRSS1, COL1A1.

## Federation

One client per **sequencing study**. Studies differ in protocol, sequencing
depth, and captured cell types. Xin contains only the four endocrine types;
Baron contains fourteen.

Single-study collections in the same archive are unsupported because they do
not provide multiple natural clients.

## Groups: none

One row is one cell, and cells are treated as independent. A label-aware random
train/test split is sound here, the only dataset in BioSilo where that's true.

## Cell-type harmonization

The same biological type is named differently across studies. Left alone, one
type becomes several classes and the clients stop being comparable.

| raw label | canonical | study |
|---|---|---|
| `pp` | gamma | Muraro |
| `duct` | ductal | Muraro |
| `mesenchymal` | stellate | Muraro |
| `PSC` | stellate | Segerstolpe |
| `activated_stellate` | stellate | Baron |
| `quiescent_stellate` | stellate | Baron |

All six aliases occur in the source labels and are covered by the optional
real-label tests.

## Non-biological annotations are excluded

Segerstolpe labels some droplets `co-expression` (39 cells, typically doublets
expressing two conflicting hormone programs) and `unclassified endocrine` (5).
These labels record unresolved annotations rather than biological cell types.

`co-expression` clears the 10-cell rare-type threshold. Explicit exclusion
reduces the default label set from 13 classes to 12.

The excluded set is a setting recorded in every partition's manifest, and
`exclude_types=()` keeps them.

`MHC class II` is not explicitly excluded because it denotes a cell population;
it is removed by the default rarity threshold.

## Preprocessing

CPM normalize → `log2(x + 1)` → min-max scale **per study**.

Per-study scaling reduces differences in sequencing depth and protocol before
training.

The protocol is inductive. Splits are established first; each study's min/max
values, the optional variance-ranked gene set, cell-type prevalence, and
shared-label set are fitted from training cells only and then applied unchanged
to test cells. Per-cell CPM and log transformation do not use other cells.

## Parameters

| | default | |
|---|---|---|
| `cohort` | `pancreas` | the only multi-study cohort |
| `shared_labels_only` | False | see the warning below |
| `min_cells_per_type` | 10 | drop rarer types |
| `exclude_types` | `("co-expression", "unclassified endocrine")` | non-biological annotations |
| `top_genes` | 0 | 0 keeps every shared gene; otherwise top-K by variance |
| `train_ratio` | 0.80 | label-stratified within each client |
| `seed` | 1 | controls the per-client split |
| `source_dir` | `<root>/_raw/scRNAseq` | extracted Zenodo archive; location only |

`shared_labels_only=True` reduces 12 classes to 4. Alpha, beta, delta, and gamma
are the only types present in all four studies.

## Storage

`npz` per client: `x`, `y`. No group ids.

## Scale

Generation on the full pancreas cohort peaks at several gigabytes of memory.
**Run it as a batch job** if your cluster caps interactive memory.

## Citations

Cite the benchmark collection and the four source studies.

> Abdelaal, T., Michielsen, L., Cats, D., Hoogduin, D., Mei, H., Reinders, M. J.
> T., & Mahfouz, A. (2019). A comparison of automatic cell identification methods
> for single-cell RNA sequencing data. *Genome Biology*, 20, 194. Data at Zenodo
> 10.5281/zenodo.3357167.

> Baron, M., Veres, A., Wolock, S. L., et al. (2016). A Single-Cell Transcriptomic
> Map of the Human and Mouse Pancreas Reveals Inter- and Intra-cell Population
> Structure. *Cell Systems*, 3(4), 346–360.e4.

> Muraro, M. J., Dharmadhikari, G., Grün, D., et al. (2016). A Single-Cell
> Transcriptome Atlas of the Human Pancreas. *Cell Systems*, 3(4), 385–394.

> Segerstolpe, Å., Palasantza, A., Eliasson, P., et al. (2016). Single-Cell
> Transcriptome Profiling of Human Pancreatic Islets in Health and Type 2
> Diabetes. *Cell Metabolism*, 24(4), 593–607.

> Xin, Y., Kim, J., Okamoto, H., et al. (2016). RNA Sequencing of Single Human
> Islet Cells Reveals Type 2 Diabetes Genes. *Cell Metabolism*, 24(4), 608–615.
