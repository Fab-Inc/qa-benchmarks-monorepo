# Model groups for decodability figures

`model_families.csv` is the default grouping table for
`scripts/plot_lkb_style_metrics.py`. It extends LKB's 33-entry table from commit
`5e67f9f` to cover all 31 models in the September 13 Kiswahili analysis. It is not a
complete registry of every model in `models.csv`; add new model IDs explicitly as the
analysis grows. The original LKB table and saved metadata snapshots are unchanged.

OpenAI groups GPT, o3/o4-mini and gpt-oss together, including models served through
third parties. Mistral includes Ministral. Claude, Gemini, Gemma, DeepSeek, Grok, Kimi,
Llama and Qwen retain their named groups. These are comparison groups, not claims that
all members share an architecture or serving provider.

Every assigned, scored model appears in the family figures, including groups with only
one model. Scores and confidence intervals are independent of these assignments. To use
another table, pass `--families-csv` with a local path or GCS URI; the plotter snapshots
that table into its new output directory.
