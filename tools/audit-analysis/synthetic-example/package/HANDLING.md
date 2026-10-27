# Handling this audit package (for the lead, not for annotators)

Give each annotator only their own folder:

- `annotators/annotator-1/`
- `annotators/annotator-2/`

Withhold until every annotator's sheet is locked: the drawn sample (it names each item's question, model and cohort,
and a question's text identifies its item), the key (`--key-out`), the inputs the package was built from, this file
and `manifest.json`. The manifest commits to the sample by its sha256 (`inputs.sample_sha256`) and to the key
(`key_sha256`).

Annotators do not open the experiments repository, its run records (dev records included) or the compute environment
until their coding ends: the records pair each question with its model, and the code redraws the sample. The key is opened only after both annotators' sheets are locked.

The real audit package is built only in the private environment and delivered to the annotators off GitHub:
nothing that pairs items with models or arms (the sample, the key, the inputs, the records) leaves the cluster before
both annotators' sheets are in.

Annotators fill a copy of their `sheet.csv`. Check a filled sheet against this package:

    uv run python -m harness.analysis.audit_package --check-sheet FILLED.csv --package <this directory> \
        --annotator <name> [--complete]
