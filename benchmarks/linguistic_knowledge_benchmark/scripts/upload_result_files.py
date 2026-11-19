"""
PLACEHOLDER -- not implemented yet.

Will push this benchmark's result files to GCS once a service account is set up for
it. Until then, upload the files by hand in the Cloud Console.

Files to upload, as written by write_result_files.py:

    results/linguistic-knowledge/LK-en/   LK-en_accuracy.csv
                                          LK-en_accuracy_metadata.json
                                          LK-en_accuracy_frontier.json
    results/linguistic-knowledge/LK-sw/   the same for LK-sw

Destination:

    gs://fab-benchmarks-external/benchmarks-individual/linguistic-knowledge/LK-en/
    gs://fab-benchmarks-external/benchmarks-individual/linguistic-knowledge/LK-sw/

Planned behaviour:
  1. Check every file exists before uploading anything, so the bucket never holds
     a mix of new and old files.
  2. Print local file -> GCS object for each file; stop there with --dry-run.
  3. Upload with fab_benchmark_publisher.publisher.upload_file.

Pending setup:
  - a service account with write access ("Storage Object Admin", to allow
    overwriting) on the fab-benchmarks-external bucket
  - how local runs authenticate as it: impersonation via
    GOOGLE_CLOUD_IMPERSONATE_SERVICE_ACCOUNT in the repo-root .env (see README
    "GCP local setup"), or a key file via GOOGLE_APPLICATION_CREDENTIALS
"""
