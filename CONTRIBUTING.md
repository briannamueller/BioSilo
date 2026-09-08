# Contributing to BioSilo

The core generation, storage, and loading code is dataset-independent. Each
module under `biosilo/datasets/` defines:

```
NAME  STORAGE  SCHEMA_VERSION  Params  label()  build()
```

`build()` yields one `ClientData` object per natural client. Add the module to
`biosilo/core/registry.py`; `biosilo/datasets/synthetic.py` is the smallest
complete example.

`build()` must use local data and must not initiate network downloads. A dataset
may optionally expose `download(params) -> Path`; the public
`biosilo.download(...)` function calls it as an explicit preparation step.

## Partition identity

Partitions are written to `<root>/<Dataset>/<label>_<hash>/`. The hash covers
output-affecting parameters and the dataset's `SCHEMA_VERSION`. Increment that
version manually when a code change alters generated values, samples, labels,
splits, or feature meanings. Do not increment it for documentation, tests, or
refactors that leave output identical.

The package version and Git commit are recorded in `manifest.json` as provenance,
not cache identity. Generation stages a complete partition beside its target,
validates it, and then installs it atomically. `overwrite=True` explicitly
rebuilds an existing identity.

## Tests

Run the dependency-free test files directly:

```bash
for t in tests/test_*.py; do python "$t"; done
```

Most tests use fixtures. Data-backed tests skip unless their documented
environment variables are set. New behavior needs a focused fixture test;
changes to download or installation paths also need an interrupted-operation
regression test.

Comments and documentation should describe the current contract and the reason
for non-obvious choices. Use version-control history for superseded behavior and
bug narratives.
