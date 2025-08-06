"""
Main CLI application for the Drupal Data Fetcher.
"""

import typer
from loguru import logger

from src.extracter import Extracter
from src.transformer import Transformer
from src.cloud import CloudManager
from src.config import (
    GCP_PROJECT_ID,
    GCP_BUCKET_NAME,
    BIGQUERY_DATASET_ID,
    PROCESSED_DATA_DIR,
    DATASET_MAPPINGS,
)

app = typer.Typer()


@app.command()
def extract(
    entity_type: str = typer.Argument(
        ..., help="Entity type to extract (e.g., project, user, event)"
    ),
    start_page: int = typer.Option(0, help="Starting page (default: 0)"),
    end_page: int = typer.Option(None, help="Ending page (default: None for all)"),
    force: bool = typer.Option(False, help="Refetch even if files exist"),
):
    """Extract JSON data from Drupal.org API."""
    extracter = Extracter()
    extract_success = extracter.pull_dataset(entity_type, start_page, end_page, force)
    if extract_success:
        logger.info(f"Successfully extracted {entity_type} data")
    else:
        logger.error(f"Failed to extract {entity_type} data")
    return extract_success


@app.command()
def transform(
    entity_type: str = typer.Argument(..., help="Entity type to transform"),
):
    """Run transform step of a pipeline for a given entity types."""
    transformer = Transformer()
    cleaned_parquet_path = transformer.process_entity(entity_type)
    return cleaned_parquet_path


@app.command()
def load(
    entity_type: str = typer.Argument(..., help="Entity type to load"),
):
    """Load processed parquet file to GCS and BigQuery."""
    if not all([GCP_PROJECT_ID, GCP_BUCKET_NAME, BIGQUERY_DATASET_ID]):
        logger.error(
            "GCP_PROJECT_ID, GCP_BUCKET_NAME, and BIGQUERY_DATASET_ID must be set in config/environment"
        )
        return

    # Find processed parquet file
    parquet_path = PROCESSED_DATA_DIR / f"{entity_type}_processed.parquet"
    if not parquet_path.exists():
        logger.error(f"Processed file not found: {parquet_path}")
        return

    logger.info(f"Loading {entity_type} from {parquet_path}")
    cloud = CloudManager(GCP_BUCKET_NAME, GCP_PROJECT_ID)
    blob_name = f"{parquet_path.name}"

    upload_success = cloud.push_to_bucket(parquet_path, blob_name)
    if upload_success:
        bq_success = cloud.insert_parquet_to_bigquery_from_gcs(
            blob_name, BIGQUERY_DATASET_ID, entity_type
        )
        if bq_success:
            logger.info(f"✓ Successfully deployed {entity_type}")
            return True
        else:
            logger.error(f"✗ Failed to load {entity_type} to BigQuery")
            return False
    else:
        logger.error(f"✗ Failed to upload {entity_type}")
        return False


@app.command()
def load_all_processed_to_bq():
    """Load all processed parquet files to BigQuery tables."""
    if not all([GCP_PROJECT_ID, GCP_BUCKET_NAME, BIGQUERY_DATASET_ID]):
        logger.error(
            "GCP_PROJECT_ID, GCP_BUCKET_NAME, and BIGQUERY_DATASET_ID must be set in config/environment"
        )
        return

    cloud = CloudManager(GCP_BUCKET_NAME, GCP_PROJECT_ID)

    parquet_files = list(PROCESSED_DATA_DIR.glob("*.parquet"))
    logger.info(
        f"Found {len(parquet_files)} processed parquet files in {PROCESSED_DATA_DIR}"
    )

    for parquet_file in parquet_files:
        # Extract table name from filename (remove file extensions and suffixes)
        table_id = (
            parquet_file.stem.replace("_processed", "")
            .replace("_merged", "")
            .replace("_compiled", "")
        )

        logger.info(
            f"Processing {parquet_file.name} -> {BIGQUERY_DATASET_ID}.{table_id}"
        )

        # Upload to bucket
        blob_name = f"{parquet_file.name}"
        upload_success = cloud.push_to_bucket(parquet_file, blob_name)

        if upload_success:
            # Load to BigQuery
            bq_success = cloud.insert_parquet_to_bigquery_from_gcs(
                blob_name, BIGQUERY_DATASET_ID, table_id
            )
            if bq_success:
                logger.info(f"✓ Successfully loaded {table_id}")
            else:
                logger.error(f"✗ Failed to load {table_id} to BigQuery")
        else:
            logger.error(f"✗ Failed to upload {parquet_file.name}")


@app.command()
def pipeline(
    entity_type: str = typer.Argument(..., help="Entity type to process"),
):
    """Run full pipeline: extract -> transform -> upload -> load to BQ."""
    if not all([GCP_PROJECT_ID, GCP_BUCKET_NAME, BIGQUERY_DATASET_ID]):
        logger.error(
            "GCP_PROJECT_ID, GCP_BUCKET_NAME, and BIGQUERY_DATASET_ID must be set in config/environment"
        )
        return

    # Extract
    logger.info(f"Starting pipeline for {entity_type}")
    extract_success = extract(entity_type)

    if not extract_success:
        logger.error(f"Failed to extract {entity_type}")
        return

    # Transform
    transform(entity_type)

    # Deapp.command()ploy to cloud
    load(entity_type)
    logger.info(f"Pipeline completed for {entity_type}")


@app.command()
def pipeline_all():
    """Run full pipeline for all entity types."""
    if not all([GCP_PROJECT_ID, GCP_BUCKET_NAME, BIGQUERY_DATASET_ID]):
        logger.error(
            "GCP_PROJECT_ID, GCP_BUCKET_NAME, and BIGQUERY_DATASET_ID must be set in config/environment"
        )
        return

    # Get all entity types from dataset mappings
    entity_types = list(DATASET_MAPPINGS.keys())

    logger.info(f"Found {len(entity_types)} entity types to process: {entity_types}")

    for entity_type in entity_types:
        logger.info(f"Starting pipeline for {entity_type}")

        # Extract
        extract_success = extract(entity_type)

        if not extract_success:
            logger.error(f"Failed to extract {entity_type}, skipping...")
            continue

        # Transform
        try:
            transform(entity_type)
        except Exception as e:
            logger.error(f"Failed to transform {entity_type}: {e}, skipping...")
            continue

        # Upload and load to app.command()BigQuery
        deploy_success = load(entity_type)
        if deploy_success:
            logger.info(f"✓ Pipeline completed successfully for {entity_type}")
        else:
            logger.error(f"✗ Failed to deploy {entity_type}")


if __name__ == "__main__":
    app()
