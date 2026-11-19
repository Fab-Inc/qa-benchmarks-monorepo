# Inputs file format, version 1

The inputs file is the interface between a benchmark and the runner. The benchmark
writes it with its own code; `fab-bench create-experiment` stores it as an experiment.
What is in the file is exactly what models see: the runner builds no prompts and adds
nothing. `tests/behaviour/test_inputs_file.py` checks each rule here.

## Example

```json
{"benchmark_item_id": "1042", "prompt": "Solve the problem and give the letter of the correct option.\nQuestion: …", "images": [{"uri": "gs://bucket/mathv/images/1042.jpg", "detail": "high"}]}
{"benchmark_item_id": "sw-0042", "prompt": "Evaluate whether …", "system_prompt": "You are …"}
```

## The file

- JSON Lines: one JSON object per line, in UTF-8 without a byte-order mark. Lines end
  with `\n` or `\r\n`; the last line's newline is optional. Blank lines are refused.
- At least one line. Line order is the order in which each model is called, and the
  order of its result rows.
- The file can be local or on GCS.

## Items

| Key                 | Type             | Required | Meaning                                                                                                                                                  |
| ------------------- | ---------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark_item_id` | string           | yes      | Identifies the item in the benchmark; unique in the file. Not empty, no leading or trailing whitespace. Numbers must be written as strings, as `"1042"`. |
| `prompt`            | string           | yes      | The user message text, sent as it is. Must hold more than whitespace.                                                                                    |
| `system_prompt`     | string or `null` | no       | The system message, sent as it is. Omitted or `null` means none; otherwise it must hold more than whitespace.                                            |
| `images`            | list of images   | no       | Sent in the item's user message, in list order. Omitted or `[]` means none.                                                                              |

Any other key is refused, so a misspelled field can't silently drop part of an input.
That includes item data the model doesn't see: see
[Not in this format](#not-in-this-format).

## Images

| Key      | Type              | Required | Meaning                                                                                      |
| -------- | ----------------- | -------- | -------------------------------------------------------------------------------------------- |
| `uri`    | string            | yes      | `gs://bucket/path`, an absolute local path (`~` is expanded), or a path relative to the file |
| `detail` | `"low"`, `"high"` | yes      | How FDLLM sizes the image when sending it                                                    |
| `sha256` | string            | no       | The file's SHA-256, 64 lowercase hex characters; creation checks the file against it         |

- A relative path resolves against the inputs file's folder, locally or on GCS. In a
  file on GCS it can't leave the bucket, and local paths are refused.
- An experiment on GCS can't use images at local paths unless they are copied into it
  with `--copy-images`.
- One image can appear in several items, and more than once in one item.
- Any file Pillow opens works; FDLLM sends it as PNG.

## What a model receives

1. The system message, if the item has one.
1. One user message: the images in order, each resized by FDLLM for its detail level
   (`low` to 512×512; `high` to at most 2048 on the long side and 768 on the short
   side), then the prompt text.

The token limit comes from the experiment, not the inputs file.

## What identifies a call

A call's `model_call_id` hashes the `benchmark_item_id`, `prompt`, `system_prompt`, and
each image's `sha256` and `detail` in order, together with the model and its registered
config. An image's URI is left out, so moving or copying images keeps every result.

## How an experiment stores it

Creation reads every image once and stores a normalised copy of the file:

- Canonical JSON: sorted keys, non-ASCII characters unescaped, `system_prompt` written
  as `null` when absent, and `images` written only when there are some.
- Each image gets its `sha256`. A local path becomes absolute; a copied image's `uri`
  becomes `images/<sha256><suffix>` inside the experiment, with the original location in
  `source_uri`.

The experiment's `inputs_sha256` is the hash of this stored copy. Two files that differ
only in key order, spacing or omitted optional keys give the same experiment. Image URIs
are part of the stored copy, so they change `inputs_sha256`, but not any call ID.

## Writing it from Python

```python
from fab_benchmark_runner import ImageRef, ModelInput, write_inputs

write_inputs(
    [
        ModelInput(
            "1042",
            "Solve the problem …",
            images=(ImageRef("gs://bucket/mathv/images/1042.jpg", "high"),),
        ),
        ModelInput("sw-0042", "Evaluate whether …", system_prompt="You are …"),
    ],
    "inputs.jsonl",
)
```

`write_inputs` checks the same rules, and `create_experiment` also takes `ModelInput`
values directly.

## Checking a file

```bash
fab-bench check-inputs inputs.jsonl
```

This checks every rule above and lists every problem at once, each with its line and,
for an image, its position in the item. It exits with status 1 if there is any:

```text
inputs.jsonl:2: unknown keys ['answer']
inputs.jsonl:2: benchmark_item_id must be a string: write the number 1042 as "1042"
inputs.jsonl:4: image 1: /data/mathv/images/1042.png does not exist
3 problems in inputs.jsonl
```

It reads every image, local or on GCS, to check that it exists and, where the file gives
a `sha256`, that the file matches it. Whether an image opens is checked when a run sends
it. A file without problems gets its counts:

```text
Items: 4
Images: 2 files, used 3 times, all found
No problems in inputs.jsonl.
```

The file can be local or a `gs://` URI. From Python, `check_inputs_file(path)` returns
the same as an `InputsCheck`. Creating an experiment checks the same rules: it lists
every problem in the rows, then stops at the first image that is missing or doesn't
match.

## Not in this format

- Item data the model doesn't see, such as answer keys, subjects or levels. A benchmark
  keeps these itself and joins result rows on `benchmark_item_id`. In the inputs they
  would count in the experiment's hash, so fixing an answer key would make a new
  experiment and rerun every model.
- Several turns, an assistant prefill, or text between images.
- Settings per item, such as a token limit or temperature.
- Documents such as PDFs.
