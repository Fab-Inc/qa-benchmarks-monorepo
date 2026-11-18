"""
PLACEHOLDER -- not implemented yet.

Will push this benchmark's result files to GCS once a service account is set up for
it. Until then, upload the files by hand in the Cloud Console.

Files to upload, as written by website_metrics_frontiers.py and
packages/fab_benchmark_publisher/scripts/write_cluster_metadata.py:

    results/linguistic-knowledge/LK-en/   3 files (metrics CSV, frontiers JSON,
    results/linguistic-knowledge/LK-sw/   benchmark_metadata.json) per language
    results/foundational-literacy_cluster_metadata.json

Destination:

    gs://fab-benchmarks-external/benchmarks-individual/linguistic-knowledge/LK-en/
    gs://fab-benchmarks-external/benchmarks-individual/linguistic-knowledge/LK-sw/
    cluster metadata: location not decided yet

Planned behaviour:
  1. Re-read and validate every file from disk (benchmark_results.
     read_benchmark_results, result_metadata.read_cluster_meta) and stop before
     uploading anything if one fails, so the bucket never holds a mix of new and
     old files.
  2. Print local file -> GCS object for each file; stop there with --dry-run.
  3. Upload with fab_benchmark_publisher.publisher.upload_file.

Pending setup:
  - a service account with write access ("Storage Object Admin", to allow
    overwriting) on the fab-benchmarks-external bucket
  - how local runs authenticate as it: impersonation via
    GOOGLE_CLOUD_IMPERSONATE_SERVICE_ACCOUNT in the repo-root .env (see README
    "GCP local setup"), or a key file via GOOGLE_APPLICATION_CREDENTIALS
  - where the cluster metadata file goes in the bucket
  - benchmark_results.gcs_object_name still builds the old
    website/benchmark-clusters/... paths; update it to benchmarks-individual/...
"""
