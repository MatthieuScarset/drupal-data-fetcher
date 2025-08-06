import ast
import pandas as pd
import typer
from loguru import logger
from typing import Optional
import re

from src.fetcher import Fetcher
from src.config import FETCHER_UPDATES_URL
from src.config import DATASET_MAPPINGS
from src.config import INTERIM_DATA_DIR

################################################################################
# Helper functions
################################################################################


def ensure_dataset_name(name: str):
    """
    Ensure the dataset name is valid.
    """
    if name not in DATASET_MAPPINGS:
        raise ValueError(
            f"Invalid dataset name: {name}. Available datasets: {', '.join(DATASET_MAPPINGS.keys())}")

    return name


def apply_extract_by_key(x, key: str):
    """
    Extracts the value of a given key from a dictionary or a string representation of a dictionary.
    If the input is not a dictionary or a string that can be evaluated to a dictionary, it returns None.

    Args:
        x: The input which can be a dictionary or a string representation of a dictionary.
        key: The key whose value needs to be extracted.

    Returns:
        The value associated with the key if found, otherwise None.
    """
    if isinstance(x, dict):
        return x.get('id')
    if isinstance(x, str):
        try:
            d = ast.literal_eval(x)
            if isinstance(d, dict):
                return d.get('id')
        except Exception:
            return None
    return None


def rename_taxonomy_columns(
    df: pd.DataFrame,
    col_name_prefix: str = 'taxonomy_vocabulary_'
):
    """
    Rename taxonomy columns in the DataFrame based on the dataset.

    Args:
        df (RawDataset): The DataFrame containing the raw dataset.
        col_name_prefix (str): The prefix for taxonomy columns to be renamed.
    """
    df_vocabularies = pd.read_csv(INTERIM_DATA_DIR / 'vocabulary_merged.csv')
    if df_vocabularies.empty:
        raise ValueError(
            "Vocabulary DataFrame is empty. Cannot rename columns.")

    columns_to_rename = {}
    for col in df.columns:
        if col.startswith(col_name_prefix):
            vid = col.split('_')[-1]
            vid_mask = df_vocabularies['vid'] == int(vid)
            vocab = df_vocabularies.loc[vid_mask, 'name']
            if not vocab.empty:
                vocab_name = vocab.astype(str).iloc[0]
                vocab_name = vocab_name.replace(' ', '_')
                vocab_name = vocab_name.lower()
                columns_to_rename[col] = vocab_name

    if columns_to_rename:
        df.rename(columns=columns_to_rename, inplace=True)


def load_large_csv_to_pd(
    input_file: str,
    chunk_size: int = typer.Option(1000)
) -> pd.DataFrame:
    chunk_list = []
    for chunk in pd.read_csv(
        input_file,
        low_memory=False,
        chunksize=chunk_size,
        on_bad_lines='skip',
        dtype=str  # Force all columns to be read as strings
    ):
        # Convert everything to string, handling all edge cases
        for col in chunk.columns:
            chunk[col] = chunk[col].apply(
                lambda x: str(x) if pd.notna(x) else '')

        chunk_list.append(chunk)

    return pd.concat(chunk_list, ignore_index=True)


def get_last_commits(fetcher: Fetcher, project_id: str, namespace: str = 'project') -> Optional[str]:
    """
    Fetch the last commit ID for a given project ID.

    Args:
        project_id: Project ID

    Returns:
        Last commit ID or None if not found
    """
    url = f"https://git.drupalcode.org/api/v4/projects/{namespace}%2F{project_id}/repository/commits"
    try:
        response = fetcher._make_request(url)
        if response is None or response.status_code != 200:
            logger.error(f"Failed to fetch commits for project {project_id}")
            return None
        return response.json()
    except Exception as e:
        logger.error(
            f"Error fetching last commit ID for project {project_id}: {str(e)}")
        return None


def get_core_compatibility(fetcher: Fetcher, field_project_machine_name: str, version: str = 'current') -> Optional[str]:
    """
    Fetch core compatibility information for a project.

    Args:
        field_project_machine_name: Project machine name
        version: Version to check (default: 'current')

    Returns:
        Core compatibility string or None if not found

    Raises:
        APIError: If request fails
    """
    url = f"{FETCHER_UPDATES_URL}/{field_project_machine_name}/{version}"

    try:
        logger.debug(f"{url}")
        response = fetcher._make_request(url)
        if response is None or response.status_code != 200:
            raise Exception(
                f"Failed to fetch data for {field_project_machine_name} from {url}")

        # Extract core compatibility using regex
        matches = re.findall(
            r'<core_compatibility>(.*?)</core_compatibility>', response.text)

        if matches:
            compatibility = matches[0]
            logger.info(f"{field_project_machine_name}: {compatibility}")
            return compatibility
        else:
            logger.warning(
                f"{field_project_machine_name}: no core compatibility found")

    except Exception as e:
        logger.error(
            f"Failed to fetch core compatibility for {field_project_machine_name}:{version}: {str(e)}")

    return None
