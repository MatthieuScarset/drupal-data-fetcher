import os
import pandas as pd
from loguru import logger
import typer
from src.config import DATASET_MAPPINGS
from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import ensure_dataset_name, rename_taxonomy_columns, load_large_csv_to_pd

app = typer.Typer()


@app.command()
def process(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., issue)"),
):
    """
    Process a specific dataset and save the cleaned DataFrame.

    Args:
        dataset (str): The name of the dataset to process.
    """
    ensure_dataset_name(dataset)

    # Load the dataset
    logger.info(f"Cleaning dataset: {dataset}")
    df = load_large_csv_to_pd(str(INTERIM_DATA_DIR / f"{dataset}_merged.csv"))
    logger.info(f"Loaded dataset: {dataset} with shape {df.shape}")

    # Rename columns based on the dataset
    logger.info("Renaming taxonomy columns...")
    rename_taxonomy_columns(df)

    cols_to_drop = [
        # General nodes' columns from Drupal
        'has_new_content',
        'flag_flag_tracker_follow_user',
        'feed_nid',
        'feeds_item_url',
        'feeds_item_guid',
        'book_ancestors',
        'sticky'
        'promote',
        'status',
        'edit_url',
        'url',
        'language',
        'is_new',
        'vid',
        # Comment related columns
        'comment_count_new',
        'comments',
    ]

    # Projects
    if dataset == "project":
        # Issue count
        if not os.path.exists(INTERIM_DATA_DIR / "issue_count.csv"):
            raise FileNotFoundError(
                'Issue count file does not exist.\nHave you run the command? make data ARGS="generate_extra"')
        issues_count = pd.read_csv(INTERIM_DATA_DIR / "issue_count.csv")
        df = df.merge(issues_count, on='nid', how='left')
        df['issue_count'] = df['issue_count'].fillna(0).astype(int)
        logger.info(f"Projects with issues: {(df['issue_count'] > 0).sum()}")
        logger.info(f"Total issues in all projects: {df['issue_count'].sum()}")

    # Issues
    elif dataset == "issue":
        # Project core compatibility per issue version.
        cols_to_drop.extend([
            # Issue related columns
            'field_issue_files',            
            'field_project_machine_name',
            'field_project_version'
        ])
        pass

    logger.info(f"Saving processed: {dataset}")
    df.to_parquet(PROCESSED_DATA_DIR / f"{dataset}_processed.parquet")

@app.command()
def process_all():
    """
    Prepare RAW data for analysis and modeling.
    """
    logger.info(f"Processing raw datasets")
    for name in DATASET_MAPPINGS.keys():
        process(dataset=name)


if __name__ == "__main__":
    app()
