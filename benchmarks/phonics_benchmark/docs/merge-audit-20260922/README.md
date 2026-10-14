# Preservation of the September unified phonics runs

The September 22 source snapshot was merged, restricted to 991 English and 1,000
Kiswahili items, and packaged as unified runs. Two Qwen models were removed because
their token limits did not match the unified runner configuration. The September 23
Astra additions were subsequently incorporated. All retained response values and their
recorded source locations match the archived source files.

## Actual response accounting

| Transformation                    | English | Kiswahili |
| --------------------------------- | ------- | --------- |
| Downloaded September 22 rows      | 62,595  | 110,611   |
| Identical copies removed          | -242    | -6,688    |
| Other duplicate responses removed | -2,156  | -112      |
| Outside selected items            | -23,753 | -63,591   |
| Two Qwen models removed           | -902    | -148      |
| September 23 Astra additions      | 368     | 452       |
| Final unified responses           | 35,910  | 40,524    |

| Final data                                           | English | Kiswahili |
| ---------------------------------------------------- | ------- | --------- |
| Items                                                | 991     | 1,000     |
| Scorable responses                                   | 35,437  | 40,250    |
| Retained failed calls                                | 380     | 253       |
| Retained parsing failures                            | 93      | 21        |
| Later responses not incorporated                     | 345     | 0         |
| Source call groups with differing successful outputs | 1,724   | 60        |

Failed and unparseable responses remain in the unified files. They are excluded from
scoring, not from preservation. Parsing exclusions above are the sealed historical
classifications; this audit does not redefine or rerun the parser. The later
unincorporated rows are outside the September 22 starting population. See the model
breakdown for their identities and counts.

## Sources and dispositions

Each September 22 source occurrence appears exactly once below. The disposition order
is: duplicate, outside selected items, excluded model, retained. September 23 rows
appear only when absent from the September 22 selection, avoiding counting the same
historical responses twice. Source runs are download destinations; a row's original run
ID can differ when earlier responses were copied there.

| Language  | Snapshot            | Source run                                | Copies | Other duplicates | Outside subset | Qwen removed | Kept  | Added | Not incorporated |
| --------- | ------------------- | ----------------------------------------- | ------ | ---------------- | -------------- | ------------ | ----- | ----- | ---------------- |
| english   | 20260922-coverage90 | en-extension-frontier-20260918            | 0      | 0                | 1701           | 0            | 9471  | 0     | 0                |
| english   | 20260922-coverage90 | en-extension-hf-20260918                  | 0      | 0                | 405            | 0            | 2255  | 0     | 0                |
| english   | 20260922-coverage90 | en-extension-or-gemma-20260918            | 0      | 0                | 243            | 0            | 1353  | 0     | 0                |
| english   | 20260922-coverage90 | en-extension-or-ministral-20260918        | 0      | 0                | 162            | 0            | 902   | 0     | 0                |
| english   | 20260922-coverage90 | en-extension-or-qwen14b-20260918          | 0      | 0                | 81             | 451          | 0     | 0     | 0                |
| english   | 20260922-coverage90 | en-extension-or-qwen8b-20260918           | 0      | 0                | 81             | 451          | 0     | 0     | 0                |
| english   | 20260922-coverage90 | en-extension-or-qwen9b-20260918           | 0      | 0                | 81             | 0            | 451   | 0     | 0                |
| english   | 20260922-coverage90 | en-stratified-astra-20260922              | 0      | 0                | 158            | 0            | 172   | 0     | 0                |
| english   | 20260922-coverage90 | en-stratified-hf-parallel-20260917        | 0      | 0                | 2690           | 0            | 2700  | 0     | 0                |
| english   | 20260922-coverage90 | en-stratified-or-gemma-20260917           | 0      | 2156             | 538            | 0            | 540   | 0     | 0                |
| english   | 20260922-coverage90 | en-stratified-or-ministral-20260917       | 0      | 0                | 1076           | 0            | 1080  | 0     | 0                |
| english   | 20260922-coverage90 | en-stratified-or-qwen9b-20260917          | 0      | 0                | 538            | 0            | 540   | 0     | 0                |
| english   | 20260922-coverage90 | en-ufli-schwa-stratified-20260915         | 0      | 0                | 13383          | 0            | 13546 | 0     | 0                |
| english   | 20260922-coverage90 | en-ufli-schwa-stratified-20260915-google  | 242    | 0                | 2616           | 0            | 2532  | 0     | 0                |
| english   | 20260923-astra      | en-stratified-astra-20260922              | 0      | 0                | 0              | 0            | 0     | 368   | 0                |
| english   | 20260923-astra      | en-ufli-schwa-stratified-20260915         | 0      | 0                | 0              | 0            | 0     | 0     | 345              |
| kiswahili | 20260922-coverage90 | sw-additional-hf-parallel-20260917        | 9      | 0                | 2753           | 0            | 1108  | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-additional-or-gemma-20260917           | 12     | 0                | 1647           | 0            | 663   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-additional-or-ministral-20260917       | 4      | 4                | 1098           | 0            | 442   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-additional-or-qwen9b-20260917          | 0      | 3                | 550            | 0            | 221   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-frontier-20260918     | 0      | 0                | 1743           | 0            | 1554  | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-hf-20260918           | 0      | 0                | 415            | 0            | 370   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-or-gemma-20260918     | 0      | 0                | 249            | 0            | 222   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-or-ministral-20260918 | 0      | 0                | 166            | 0            | 148   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-or-qwen14b-20260918   | 0      | 0                | 83             | 74           | 0     | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-or-qwen8b-20260918    | 0      | 0                | 83             | 74           | 0     | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-linear-extension-or-qwen9b-20260918    | 0      | 0                | 83             | 0            | 74    | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-original-astra-20260922                | 4      | 0                | 523            | 0            | 252   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-original-hf-parallel-20260917          | 0      | 11               | 4792           | 0            | 3522  | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-original-or-gemma-20260917             | 4994   | 0                | 0              | 0            | 1     | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-original-or-ministral-20260917         | 1665   | 0                | 960            | 0            | 705   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-original-or-qwen9b-20260917            | 0      | 1                | 959            | 0            | 705   | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-tusome-clusters-20260905               | 0      | 12               | 9591           | 0            | 7047  | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-tusome-clusters-20260906-remaining     | 0      | 81               | 23580          | 0            | 17219 | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-tusome-clusters-additional-20260916-a  | 0      | 0                | 6974           | 0            | 2824  | 0     | 0                |
| kiswahili | 20260922-coverage90 | sw-tusome-clusters-additional-20260916-b  | 0      | 0                | 7342           | 0            | 2995  | 0     | 0                |
| kiswahili | 20260923-astra      | sw-original-astra-20260922                | 0      | 0                | 0              | 0            | 0     | 452   | 0                |

[Source table](responses_by_source.csv) ·
[Source and model table](responses_by_source_and_model.csv) ·
[Input hashes](input_hashes.csv) · [Verification](verification.json)

## Checks and reproduction

The audit reads the archived JSONL files independently of the merging helpers. It checks
file hashes, the complete source item union, exact response-ID sets and model/item
uniqueness, every provenance occurrence, and the selected duplicate winner (success,
then latest timestamp, then source path and line). It compares all retained response
fields; only equivalent integral JSON numbers such as 30000 and 30000.0 are normalized.
Corrected labels are checked against the sealed corrected dataset; their linguistic
correctness is outside this preservation check.

The full response files survive the gold refresh unchanged, the selection contains
exactly its declared items, and the final item values match that corrected selection.
The Astra additions preserve every existing response. All inputs are hash-checked again
after the audit. No cloud access or model calls are involved.

Run from the repository root with the archived folders under `outputs/`:

```sh
python benchmarks/phonics_benchmark/scripts/audit_unified_preservation.py \
  --report-dir outputs/merge-audit-review \
  --ledger outputs/merge-audit-review-rows.csv.gz
```

Use `--outputs-root` to relocate the archives. Existing reports and ledgers are refused.
The compressed ledger records every source row's location, call ID, content hash,
disposition and selected occurrence. It remains under ignored `outputs/`; the small
tables and verification record are kept with this document. The archived source files
are required to rerun this check; normal CI runs the small metric examples and audit
rejection tests without downloading the archives.
