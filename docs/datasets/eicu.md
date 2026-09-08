# eICU

ICU outcome prediction from the eICU Collaborative Research Database, federated
by hospital. Over 200,000 ICU stays from 208 US hospitals, one of the few large
public clinical datasets with institution identifiers.

## Getting the data

Two source distributions are available:

**Demo (open).** ~2,500 stays from 186 hospitals, no credentials needed.
Distributed under ODbL. Enough to run the whole pipeline and check it works.

**Full (credentialed).** Requires:

1. A [PhysioNet](https://physionet.org/) account
2. Completed CITI training
3. An accepted data use agreement for the
   [eICU-CRD](https://physionet.org/content/eicu-crd/2.0/)

Approval takes days to weeks. A full build has reached about 105 GB of virtual
memory in testing, so request at least 128 GB and use a batch node. The demo
needs about 3 GB.

BioSilo does not redistribute either source. Extract the tables under
`data/_raw/eICU/`, or pass `source_dir` if they are stored elsewhere.

```python
import biosilo

path = biosilo.generate(
    "eICU", task="mortality_24h",
)
data = biosilo.load("eICU", partition=path.name)
```

## Task

Binary classification, chosen per run. Because these outcomes are imbalanced,
balanced accuracy is preferable to unadjusted accuracy.

| task | window | meaning |
|---|---|---|
| `mortality_24h` | 24 h | dies in hospital |
| `mortality_48h` | 48 h | dies in hospital |
| `los_3day` | 24 h | total ICU stay exceeds 3 days |
| `los_7day` | 24 h | total ICU stay exceeds 7 days |
| `shock_4h` | 4 h | a vasopressor infusion begins after the window |
| `shock_12h` | 12 h | as above |
| `arf_4h` | 4 h | a respiratory-support proxy begins after the window |
| `arf_12h` | 12 h | as above |

**Shock** uses the first positive-rate vasopressor infusion documented in
`infusionDrug` (norepinephrine, epinephrine, dopamine, vasopressin, or
phenylephrine). `medication` is not used because it records orders, which do
not prove administration.

**ARF** is an operational respiratory-support proxy, not a diagnosis. Its onset
is the earliest non-zero mechanical-ventilation start offset, positive PEEP
limit, or PEEP/CPAP charting in `respiratoryCharting`.

Negative ventilation-start offsets are retained because they denote support
that began before ICU admission; those stays are excluded from prospective
event prediction. A zero start offset is treated as unavailable because eICU
uses zero for missing date/time-derived offsets in this table.

These are operational chart-derived endpoints. A negative label means no
qualifying event was documented, not proof that the clinical condition never
occurred. Source-table availability and charting practice vary by hospital and
should be considered when interpreting cross-hospital results.

Patients whose event occurs at or before the end of the prediction window are
excluded from event tasks. This includes an onset exactly at 4 or 12 hours,
because that timestamp belongs to the last modeled hour.

The LOS tasks use the first 24 hours as input. Their names identify the outcome
threshold, not the input duration. Using 72 or 168 hours of input would reveal
that the patient had already reached the corresponding LOS threshold.

## Sample shape

Two inputs:

```
ts     (T, n_ts)      hourly clinical time series; T is the task's window
static (n_static,)    admission features and diagnosis codes, no time axis
```

`T` is fixed. Stays are truncated to the window, and patients without a complete
window are dropped so sequence length cannot encode an early outcome.

The time-series channels are `[values | masks]`: the first half are values, the
second half encode whether and how recently each value was measured. With
`mask_mode="exponential_decay"`, an observed value has mask 1, hours before its
first observation have mask 0, and later missing hours use
`1 / max(missing_hours * decay_rate, 1)`.

## Federation

By `hospitalid`. Hospitals differ in size, case mix, and charting practice,
which is the heterogeneity the benchmark exists to capture.

## Groups: person

The row unit is `patientunitstayid`, an ICU **stay**. A readmitted patient
contributes several rows that share chronic diagnoses, baseline labs, and
physiology, so a split between them leaks. BioSilo emits `uniquepid` as the
group id.

In the demo cohort, **416 of 1,841 people have more than one ICU stay**.

## Pipeline

**Stage 1: raw tables to a feature store.** Each eligible stay has exactly one
row for every positive hour through discharge (capped at 14 days). Lab,
respiratory, nursing, periodic, and aperiodic features are joined on
`(patient, hour)`; an hour with no measurements remains as an all-missing raw
row. Known FiO2 and temperature units are normalized before hourly averaging.
Exact feature names appearing in multiple sources stay source-qualified rather
than being silently averaged; semantically similar names remain separate.

The shared cohort contains adults with ICU stays longer than five hours and at
least one measurement in the five time-series sources. A missing mortality
outcome excludes a stay only from mortality tasks; it does not remove a valid
LOS or event-task sample.

People are assigned globally to train or test before fitted preprocessing.
Feature prevalence, 5/95 percentile normalization, demographic scaling and
categorical vocabularies, and diagnosis vocabularies use training stays only;
time-series fitting also stops at the task's observation horizon. Masks and
forward filling reset at every patient. The fitted transform is then applied to
both partitions. Diagnosis events are restricted to the requested diagnosis
window and can never extend beyond the task's observation horizon.

BioSilo-generated stage-one stores are therefore keyed by task horizon, train
ratio, seed, and preprocessing parameters under `data/_cache/eICU/`. If an
explicit `cache_dir` contains an external store, it is used as-is and recorded as `external-unverified`
because its preprocessing protocol cannot be established.

**Stages 2–3: feature store to clients.** Derive labels, apply the observation
window, group by hospital, filter and rank hospitals, and preserve the global
person assignment inside every client. One person cannot be training data at
one hospital and test data at another.

## Parameters

| | default | |
|---|---|---|
| `task` | `mortality_24h` | see table above |
| `num_clients` | 20 | maximum clients; 0 keeps every qualifying hospital; a shortfall is reported |
| `sort_mode` | `size` | `size`, `positives`, or `prevalence` |
| `min_size` | 10 | minimum stays per hospital |
| `min_prev` | 0.0 | minimum positive rate |
| `min_minority` | 0 | minimum minority-class count |
| `train_ratio` | 0.75 | global person-level training fraction |
| `seed` | 1 | controls the global person assignment |
| `include_diagnoses` | True | append diagnosis features to `static` |
| `diag_window` | `5h` | diagnosis inclusion window, from `1h` to `5h`; capped at the task's input window |
| `paradigm` | `single_horizon` | the only value; see below |
| `drop_hospital_vars` | True | drop columns describing the *hospital*, see below |
| `min_dx_prevalence` | 0.01 | stage 1 |
| `within_prev` / `cross_prev` | 0.25 / 0.70 | stage 1 double-threshold filter |
| `mask_mode` / `decay_rate` | `exponential_decay` / 4⁄3 | stage 1 |
| `source_dir` | `<root>/_raw/eICU` | extracted source tables; location only |
| `cache_dir` | `<root>/_cache/eICU` | reusable feature stores; location only |

`drop_hospital_vars` removes teaching status, bed-count category, region, and
physician speciality to prevent direct site identification from these fields.

`paradigm` is reserved. Only `single_horizon` is implemented, producing one
label per stay with `y` shape `(N,)`. Rolling labels would require a separate
`(N,T)` output contract.

## Storage

`npz` per client: `ts`, `static`, `y`, and group ids.

## Citations

Cite the database.

> Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., &
> Badawi, O. (2018). The eICU Collaborative Research Database, a freely
> available multi-center database for critical care research. *Scientific Data*,
> 5, 180178.
