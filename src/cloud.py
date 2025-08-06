"""Cloud operations for GCP bucket and BigQuery integration."""

from pathlib import Path
import time

from google.cloud import bigquery, storage
from google.api_core import retry
from loguru import logger


class CloudManager:
    """Manages GCP bucket operations and BigQuery integration."""

    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)

    def push_to_bucket(
        self,
        local_path: Path,
        blob_name: str,
        timeout: int = 300,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> bool:
        """Push file to GCP bucket with chunked upload and retry logic."""
        file_size = local_path.stat().st_size
        max_retries = 3

        # Use chunked upload for files larger than 32MB
        use_chunked = file_size > 32 * 1024 * 1024

        logger.info(
            f"Uploading {local_path} ({file_size / (1024 * 1024):.1f}MB) to gs://{self.bucket_name}/{blob_name}"
        )
        logger.info(f"Using {'chunked' if use_chunked else 'simple'} upload method")

        for attempt in range(max_retries):
            try:
                blob = self.bucket.blob(blob_name)

                if use_chunked:
                    # Set chunk size for resumable uploads
                    blob.chunk_size = chunk_size

                    # Use resumable upload with timeout and retry
                    with open(local_path, "rb") as file_obj:
                        blob.upload_from_file(
                            file_obj,
                            timeout=timeout,
                            checksum="md5",
                            retry=retry.Retry(deadline=timeout),
                        )
                else:
                    # Simple upload for smaller files
                    blob.upload_from_filename(str(local_path), timeout=timeout)

                logger.info(f"✓ Upload successful on attempt {attempt + 1}")
                logger.info(
                    f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}"
                )
                return True

            except Exception as e:
                logger.warning(
                    f"Upload attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to upload {local_path} after {max_retries} attempts"
                    )

        return False

    def pull_from_bucket(self, blob_name: str, local_path: Path) -> bool:
        """Pull file from GCP bucket."""
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
            logger.info(
                f"Downloaded gs://{self.bucket_name}/{blob_name} to {local_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            return False

    def insert_parquet_to_bigquery_from_gcs(
        self, gcs_blob_name: str, dataset_id: str, table_id: str
    ) -> bool:
        """Load Parquet file from Cloud Storage directly to BigQuery table."""
        try:
            # Ensure dataset exists first
            self.ensure_bigquery_dataset_exists(dataset_id)

            table_ref = self.bq_client.dataset(dataset_id).table(table_id)
            gcs_uri = f"gs://{self.bucket_name}/{gcs_blob_name}"

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                autodetect=True,
            )

            # Load directly from Cloud Storage URI
            load_job = self.bq_client.load_table_from_uri(
                gcs_uri, table_ref, job_config=job_config
            )

            load_job.result()  # Wait for the job to complete
            logger.info(f"Loaded {gcs_uri} to {dataset_id}.{table_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {gcs_uri} to BigQuery: {e}")
            return False

    def insert_parquet_to_bigquery(
        self, parquet_path: Path, dataset_id: str, table_id: str
    ) -> bool:
        """Insert Parquet file to BigQuery table from local file."""
        try:
            # Ensure dataset exists first
            self.ensure_bigquery_dataset_exists(dataset_id)

            table_ref = self.bq_client.dataset(dataset_id).table(table_id)

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                autodetect=True,
            )

            with open(parquet_path, "rb") as source_file:
                job = self.bq_client.load_table_from_file(
                    source_file, table_ref, job_config=job_config
                )

            job.result()  # Wait for the job to complete
            logger.info(f"Loaded {parquet_path} to {dataset_id}.{table_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {parquet_path} to BigQuery: {e}")
            return False

    def ensure_bigquery_dataset_exists(self, dataset_id: str) -> bool:
        """Create BigQuery dataset if it doesn't exist."""
        try:
            self.bq_client.get_dataset(dataset_id)
            logger.debug(f"Dataset {dataset_id} already exists")
            return True
        except Exception:
            try:
                dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
                dataset.location = "US"  # Change to your preferred location
                dataset.description = "Drupal.org data repository"

                dataset = self.bq_client.create_dataset(dataset, timeout=30)
                logger.info(f"Created BigQuery dataset: {dataset_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create dataset {dataset_id}: {e}")
                return False
