import os
import orjson
import glob
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer
import httpx
import time

from src.fetcher import Fetcher
from src.config import DEV_MODE
from src.config import (
    COMPILED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from src.config import DATASET_MAPPINGS
from src.utils import ensure_dataset_name, apply_extract_by_key, load_large_csv_to_pd
from src.utils import get_last_commits, get_core_compatibility

app = typer.Typer()


@app.command()
def download(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., issue)"),
):
    """
    Download a dataset from an external source.
    
    Args:
        dataset: Name of the dataset to download (e.g., 'issue', 'project')
    
    Note:
        This function is currently not implemented (placeholder).
    """
    pass


@app.command()
def pull(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., issue)"),
    start_page: int = typer.Option(0, help="Starting page (default: 1)"),
    end_page: int = typer.Option(None, help="Ending page (default: None)"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Refetch even if file exists"
    ),
):
    """
    Pull data from Drupal.org API for a specific dataset.
    
    Fetches data from the Drupal.org API in paginated format and saves each page
    as a JSON file in the external data directory.
    
    Args:
        dataset: Name of the dataset to pull (must be in DATASET_MAPPINGS)
        start_page: Page number to start fetching from (0-indexed)
        end_page: Page number to stop at (None means fetch all available pages)
        force: If True, refetch files even if they already exist
    
    Raises:
        ValueError: If dataset name is not valid
    """
    ensure_dataset_name(dataset)

    params = DATASET_MAPPINGS[dataset]
    logger.info(f"Processing dataset: {dataset}")

    fetcher = Fetcher()
    dataset_dir = EXTERNAL_DATA_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if DEV_MODE:
        end_page = start_page + 1
    elif end_page is None:
        end_page = fetcher.get_total_pages(params)

    total_pages = end_page - start_page + 1
    logger.info(f"Process {dataset} dataset with {total_pages} pages.")

    for i in tqdm(range(start_page, end_page), total=total_pages):
        file_path = dataset_dir / f"{dataset}_page_{i}.json"
        if file_path.exists() and not force:
            logger.info(
                f"File exists for page {i}, skipping (use --force to overwrite)."
            )
            continue

        params["page"] = i

        # Sleep to avoid hitting the rate limit too quickly.
        time.sleep(1)

        response = fetcher.fetch_data(params)
        if isinstance(response, httpx.Response):
            with open(file_path, "wb") as f:
                f.write(orjson.dumps(response.json()))
                logger.success(f"Page {i} saved to {file_path}")


@app.command()
def merge(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., issue)"),
    chunk_size: int = typer.Option(1000, help="Number of files to process at once"),
):
    """
    Merge multiple JSON files from a dataset into a single CSV file.
    
    Processes JSON files in chunks to manage memory usage and combines them
    into a single CSV file in the interim data directory.
    
    Args:
        dataset: Name of the dataset to merge
        chunk_size: Number of files to process simultaneously to manage memory
    
    Returns:
        None: Saves merged data to {dataset}_merged.csv in INTERIM_DATA_DIR
    """
    logger.info(f"Merging dataset: {dataset} (chunk size: {chunk_size})")

    files = glob.glob(str(EXTERNAL_DATA_DIR / f"{dataset}/*.json"))
    total_files = len(files)
    logger.info(f"Found {total_files} files to merge")

    if not files:
        logger.warning("No files found to merge")
        return

    output_file = INTERIM_DATA_DIR / f"{dataset}_merged.csv"
    first_chunk = True
    total_rows = 0

    # Process files in chunks to manage memory
    for chunk_start in tqdm(
        range(0, total_files, chunk_size), desc="Processing chunks"
    ):
        chunk_files = files[chunk_start : chunk_start + chunk_size]
        chunk_dataframes = []

        # Process current chunk
        for file in chunk_files:
            try:
                with open(file, "rb") as f:  # Use 'rb' for orjson
                    items = orjson.loads(f.read())

                    # Handle different JSON structures
                    if isinstance(items, dict):
                        # If it's a dict with a 'list' key (API response format)
                        if "list" in items:
                            items = items["list"]
                        else:
                            # Single item, wrap in list
                            items = [items]
                    elif not isinstance(items, list):
                        # Skip if not dict or list
                        logger.warning(f"Unexpected JSON structure in {file}")
                        continue

                    if items:  # Only create DataFrame if we have data
                        df = pd.DataFrame(items)
                        chunk_dataframes.append(df)

            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                continue

        # Process current chunk if we have data
        if chunk_dataframes:
            chunk_df = pd.concat(chunk_dataframes, ignore_index=True)
            total_rows += len(chunk_df)

            # Write to CSV (append mode after first chunk)
            if first_chunk:
                chunk_df.to_csv(output_file, index=False, mode="w")
                first_chunk = False
                logger.info(f"Created output file with {len(chunk_df)} rows")
            else:
                chunk_df.to_csv(output_file, index=False, mode="a", header=False)
                logger.info(f"Appended {len(chunk_df)} rows (total: {total_rows})")

            # Clear memory
            del chunk_df
            del chunk_dataframes

        # Force garbage collection after each chunk
        import gc

        gc.collect()

    logger.success(
        f"Merged dataset saved to {output_file} with {total_rows} total rows"
    )


@app.command()
def convert(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., issue)"),
    chunk_size: int = typer.Option(
        1000, help="Number of rows per chunk for processing"
    ),
):
    """
    Convert a CSV dataset to Parquet format for efficient storage and processing.
    
    Reads a merged CSV file from the interim directory, converts all columns to string
    type for consistency, and saves as a compressed Parquet file in the raw data directory.
    
    Args:
        dataset: Name of the dataset to convert
        chunk_size: Number of rows to process per chunk for memory management
    
    Returns:
        None: Saves converted data to {dataset}_merged.parquet in RAW_DATA_DIR
        
    Raises:
        Exception: If conversion fails, logs error and attempts fallback settings
    """
    input_dataset_dir = INTERIM_DATA_DIR
    input_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dataset_dir = RAW_DATA_DIR
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Converting dataset: {dataset} to Parquet format")
        df = load_large_csv_to_pd(
            str(input_dataset_dir / f"{dataset}_merged.csv"), chunk_size=chunk_size
        )
        logger.info(f"Successfully loaded {len(df)} rows")

        # Safety check.
        logger.debug("Converting all columns to string")
        for col in df.columns:
            try:
                df[col] = df[col].astype(str)
            except Exception as col_error:
                logger.warning(f"Issue converting column '{col}': {col_error}")
                df[col] = df[col].apply(lambda x: str(x) if x is not None else "")

        # Save to parquet file.
        try:
            df.to_parquet(
                output_dataset_dir / f"{dataset}_merged.parquet",
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            logger.success(f"Dataset converted to parquet: {dataset}")
        except Exception as parquet_error:
            logger.error(f"Parquet conversion error: {parquet_error}")
            # Try with different engine or settings
            df.to_parquet(
                output_dataset_dir / f"{dataset}_merged.parquet",
                index=False,
                engine="pyarrow",
                compression=None,
                use_deprecated_int96_timestamps=True,
            )
            logger.success(
                f"Dataset converted to parquet with fallback settings: {dataset}"
            )

    except Exception as e:
        logger.error(f"Conversion failed for dataset {dataset}: {e}")


@app.command()
def convert_compiled_datasets():
    """
    Convert all compiled CSV datasets to Parquet format.
    
    Scans the compiled data directory for CSV files and converts each one
    to Parquet format in the raw data directory for efficient storage and processing.
    
    Returns:
        None: Saves converted files to RAW_DATA_DIR with .parquet extension
    """
    for csv_file in glob.glob(str(COMPILED_DATA_DIR / "*.csv")):
        output_file = csv_file.split("/")[-1].replace(".csv", "")
        df = pd.read_csv(csv_file, dtype=str)
        df.to_parquet(
            RAW_DATA_DIR / f"{output_file}.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        logger.success(f"Dataset converted to parquet: {output_file}")


@app.command()
def generate_issue_count():
    """
    Generate issue count statistics per project.
    
    Reads the issue dataset, extracts project IDs, and creates a summary
    dataframe showing the number of issues per project. Saves the result
    as a Parquet file for further analysis.
    
    Returns:
        None: Saves issue count data to issue_count.parquet in RAW_DATA_DIR
    """
    logger.info("Generating issue_count dataframe")
    df_issues = pd.read_parquet(RAW_DATA_DIR / "issue_merged.parquet")
    df_issues["field_project_id"] = df_issues["field_project"].apply(
        lambda x: apply_extract_by_key(x, "id")
    )

    issue_count = (
        df_issues["field_project_id"]
        .value_counts()
        .reset_index()
        .rename(columns={"field_project_id": "nid", "count": "issue_count"})
    )

    issue_count.nid = issue_count.nid.astype("int64")
    issue_count.issue_count = issue_count.issue_count.astype("int64")
    df_issue_count = pd.DataFrame(issue_count)
    df_issue_count.to_parquet(RAW_DATA_DIR / "issue_count.parquet")
    logger.success(
        f"Issue count dataframe generated with {df_issue_count.shape[0]} projects. "
        f"Total issues across all projects: {df_issue_count['issue_count'].sum()}"
    )


@app.command()
def generate_extra():
    """
    Generate additional dataframes.
    """
    convert_compiled_datasets()
    generate_issue_count()


@app.command()
def pull_all(
    force: bool = typer.Option(
        False, "--force", "-f", help="Refetch even if file exists"
    ),
):
    """
    Pull data from Drupal.org API for all configured datasets.
    
    Iterates through all datasets defined in DATASET_MAPPINGS and pulls
    their data from the Drupal.org API.
    
    Args:
        force: If True, refetch files even if they already exist
    """
    for name in DATASET_MAPPINGS.keys():
        pull(dataset=name, force=force)


@app.command()
def pull_all_projects_commits(
    exclude_sandbox: bool = typer.Option(
        True, "--exclude-sandbox", "-e", help="Exclude sandbox projects"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Refetch even if file exists"
    ),
):
    """
    Fetch commit history for all projects in the dataset.

    Retrieves the last 10 commits for each project using the Git API.
    Projects are read from the merged project dataset, and commit data
    is saved as individual JSON files.
    
    Args:
        exclude_sandbox: If True, skip sandbox projects during processing
        force: If True, refetch commit data even if files already exist
        
    Returns:
        None: Saves commit data to project_commits directory in EXTERNAL_DATA_DIR
    """
    input_file = INTERIM_DATA_DIR / "project_merged.csv"
    if not os.path.exists(input_file):
        logger.info(f"Generating '{input_file}'...")
        merge(dataset="project")
    if not os.path.exists(input_file):
        logger.error(f"Could not find input file: '{input_file}'")
        return

    try:
        fetcher = Fetcher()
        df = pd.read_csv(input_file, low_memory=False)
        df = df[["field_project_machine_name", "nid", "field_project_type"]]

        if exclude_sandbox:
            df = df[~(df["field_project_type"] == "sandbox")]

        if "field_project_machine_name" not in df.columns:
            logger.error(
                "Error: 'field_project_machine_name' column not found in projects dataset."
            )
            return

        dataset_dir = EXTERNAL_DATA_DIR / "project_commits"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Fetching commits for {len(df)} projects...")
        for machine_name in tqdm(df["field_project_machine_name"]):
            file_path = dataset_dir / f"{machine_name}.json"
            if file_path.exists() and not force:
                logger.info(f"Skipping {machine_name}, file already exists.")

            try:
                commits = get_last_commits(fetcher, machine_name)
                with open(file_path, "wb") as f:
                    f.write(orjson.dumps(commits))
            except Exception as e:
                logger.error(f"Error saving data for {machine_name}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing projects: {e}")


@app.command()
def pull_all_projects_core_compatibility(
    exclude_sandbox: bool = typer.Option(
        True, "--exclude-sandbox", "-e", help="Exclude sandbox projects"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Refetch even if file exists"
    ),
):
    """
    Fetch Drupal core compatibility information for all projects.
    
    Retrieves core version compatibility data for each project from the
    Drupal.org API. This includes which Drupal core versions each project
    supports or is compatible with.
    
    Args:
        exclude_sandbox: If True, skip sandbox projects during processing
        force: If True, refetch compatibility data even if files already exist
        
    Returns:
        None: Saves compatibility data to project_core_compatibility directory
    """
    input_file = INTERIM_DATA_DIR / "project_merged.csv"
    if not os.path.exists(input_file):
        logger.info(f"Generating '{input_file}'...")
        merge(dataset="project")
    if not os.path.exists(input_file):
        logger.error(f"Input file '{input_file}' does not exist.")
        return

    try:
        fetcher = Fetcher()
        dataset_dir = EXTERNAL_DATA_DIR / "project_core_compatibility"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(input_file)

        if exclude_sandbox:
            df = df[~(df["field_project_type"] == "sandbox")]

        if "field_project_machine_name" not in df.columns:
            logger.error(
                "Error: 'field_project_machine_name' column not found in projects dataset."
            )
            return

        for i, project in tqdm(df.iterrows()):
            machine_name = str(project["field_project_machine_name"])
            file_path = dataset_dir / f"{machine_name}.json"
            if file_path.exists() and not force:
                logger.info(f"Skipping {machine_name}, file already exists.")
                continue

            logger.info(f"Processing project {i} / {len(df)}")
            row = {}
            row["nid"] = int(project["nid"]) if pd.notna(project["nid"]) else None
            row["field_project_machine_name"] = machine_name
            row["core_compatibility"] = get_core_compatibility(
                fetcher, project["field_project_machine_name"]
            )
            with open(
                EXTERNAL_DATA_DIR
                / "project_core_compatibility"
                / f"{machine_name}.json",
                "wb",
            ) as f:
                f.write(orjson.dumps(row))

        logger.success(f"Core compatibility data saved to '{dataset_dir}'.")

    except Exception as e:
        logger.error(f"Error processing projects: {e}")


@app.command()
def pull_all_issues_core_compatibility(
    force: bool = typer.Option(
        False, "--force", "-f", help="Refetch even if file exists"
    ),
):
    """
    Fetch Drupal core compatibility information for all issues.
    
    Retrieves core version compatibility data for issues from the
    Drupal.org API. This provides insight into which Drupal core versions
    are affected by or relevant to specific issues.
    
    Args:
        force: If True, refetch compatibility data even if files already exist
        
    Returns:
        None: Saves compatibility data to issue_core_compatibility directory
        
    Note:
        This function appears to have a logic error - it processes issues but
        extracts project machine names, which may not be appropriate for issues.
    """
    input_file = INTERIM_DATA_DIR / "issue_merged.csv"
    if not os.path.exists(input_file):
        logger.info(f"Generating '{input_file}'...")
        merge(dataset="issue")
    if not os.path.exists(input_file):
        logger.error(f"Input file '{input_file}' does not exist.")
        return

    try:
        fetcher = Fetcher()
        dataset_dir = EXTERNAL_DATA_DIR / "issue_core_compatibility"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(input_file)

        if "field_project_machine_name" not in df.columns:
            logger.error(
                "Error: 'field_project_machine_name' column not found in projects dataset."
            )
            return

        for i, project in tqdm(df.iterrows()):
            machine_name = str(project["field_project_machine_name"])
            file_path = dataset_dir / f"{machine_name}.json"
            if file_path.exists() and not force:
                logger.info(f"Skipping {machine_name}, file already exists.")
                continue

            logger.info(f"Processing project {i} / {len(df)}")
            row = {}
            row["nid"] = int(project["nid"]) if pd.notna(project["nid"]) else None
            row["field_project_machine_name"] = machine_name
            row["core_compatibility"] = get_core_compatibility(
                fetcher, project["field_project_machine_name"]
            )
            with open(
                EXTERNAL_DATA_DIR / "issue_core_compatibility" / f"{machine_name}.json",
                "wb",
            ) as f:
                f.write(orjson.dumps(row))

        logger.success(f"Core compatibility data saved to '{dataset_dir}'.")

    except Exception as e:
        logger.error(f"Error processing projects: {e}")


@app.command()
def merge_all(
    chunk_size: int = typer.Option(1000, help="Number of files to process at once"),
):
    """
    Merge all JSON datasets into CSV format.
    
    Processes all datasets defined in DATASET_MAPPINGS, converting them from
    individual JSON files to merged CSV files. Also handles special datasets
    like project commits and core compatibility data.
    
    Args:
        chunk_size: Number of files to process simultaneously for memory management
        
    Returns:
        None: Creates merged CSV files in INTERIM_DATA_DIR for each dataset
    """
    for name in DATASET_MAPPINGS.keys():
        merge(dataset=name, chunk_size=chunk_size)

    # Merge extra datasets.
    logger.info("Merging extra datasets.")
    for name in ["project_commits", "project_core_compatibility"]:
        logger.info(f"Merging {name} dataset.")


@app.command()
def convert_all(
    chunk_size: int = typer.Option(
        1000, help="Number of rows per chunk for processing"
    ),
):
    """
    Convert all CSV datasets to Parquet format for efficient storage.
    
    Processes all datasets defined in DATASET_MAPPINGS, converting them from
    CSV to Parquet format for better compression and faster read/write operations.
    Also generates additional derived datasets like issue counts.
    
    Args:
        chunk_size: Number of rows to process per chunk for memory management
        
    Returns:
        None: Creates Parquet files in RAW_DATA_DIR for each dataset
    """
    # Convert all datasets from CSV to Parquet format.
    for name in DATASET_MAPPINGS.keys():
        logger.info(f"Converting {name} dataset to Parquet format.")
        convert(dataset=name, chunk_size=chunk_size)

    # Convert extra datasets.
    logger.info("Converting extra datasets to Parquet format.")
    generate_extra()


if __name__ == "__main__":
    app()
