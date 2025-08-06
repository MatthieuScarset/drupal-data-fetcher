import orjson
import time
from typing import Union
from loguru import logger
from tqdm import tqdm
import httpx

from src.fetcher import Fetcher
from src.config import EXTERNAL_DATA_DIR, DATASET_MAPPINGS, DEV_MODE


class Extracter:
    """
    Extracter to source paginated JSON files from Drupal.org.
    """

    def __init__(self):
        self.fetcher = Fetcher()
        self.external_dir = EXTERNAL_DATA_DIR
        self.external_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dataset_name(self, name: str):
        """
        Ensure the dataset name is valid.
        """
        if name not in DATASET_MAPPINGS:
            raise ValueError(
                f"Invalid dataset name: {name}. Available datasets: {', '.join(DATASET_MAPPINGS.keys())}"
            )

        return name

    def pull_dataset(
        self,
        dataset: str,
        start_page: int = 0,
        end_page: Union[int, None] = None,
        force: bool = False,
    ) -> bool:
        """
        Pull data from Drupal.org API for a specific dataset.

        Fetches data from the Drupal.org API in paginated format and saves each page
        as a JSON file in the external data directory.

        Args:
            dataset: Name of the dataset to pull (must be in DATASET_MAPPINGS)
            start_page: Page number to start fetching from (0-indexed)
            end_page: Page number to stop at (None means fetch all available pages)
            force: If True, refetch files even if they already exist

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.ensure_dataset_name(dataset)
            params = DATASET_MAPPINGS[dataset]
            logger.info(f"Processing dataset: {dataset}")

            dataset_dir = self.external_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)

            if DEV_MODE is True:
                end_page = start_page + 1
            elif end_page is None:
                end_page = self.fetcher.get_total_pages(params)

            total_pages = end_page - start_page
            logger.info(f"Process {dataset} dataset with {total_pages} pages.")

            for i in tqdm(range(start_page, end_page), total=total_pages):
                file_path = dataset_dir / f"{dataset}_page_{i}.json"
                if file_path.exists() and not force:
                    logger.debug(
                        f"File exists for page {i}, skipping (use force=True to overwrite)."
                    )
                    continue

                params["page"] = i

                # Sleep to avoid hitting the rate limit too quickly.
                time.sleep(1)

                response = self.fetcher.fetch_data(params)
                if isinstance(response, httpx.Response):
                    with open(file_path, "wb") as f:
                        f.write(orjson.dumps(response.json()))
                        logger.success(f"Page {i} saved to {file_path}")
                else:
                    logger.error(f"Failed to fetch page {i} for dataset {dataset}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to pull dataset {dataset}: {e}")
            return False

    def pull_all_datasets(self, force: bool = False) -> bool:
        """
        Pull data from Drupal.org API for all configured datasets.

        Args:
            force: If True, refetch files even if they already exist

        Returns:
            bool: True if all datasets were successfully pulled
        """
        success_count = 0
        total_datasets = len(DATASET_MAPPINGS)

        for dataset_name in DATASET_MAPPINGS.keys():
            logger.info(f"Pulling dataset: {dataset_name}")
            if self.pull_dataset(dataset_name, force=force):
                success_count += 1
            else:
                logger.error(f"Failed to pull dataset: {dataset_name}")

        logger.info(f"Successfully pulled {success_count}/{total_datasets} datasets")
        return success_count == total_datasets

    def get_dataset_info(self, dataset: str) -> dict:
        """Get information about a dataset including total pages and configuration."""
        try:
            self.ensure_dataset_name(dataset)
            params = DATASET_MAPPINGS[dataset]
            total_pages = self.fetcher.get_total_pages(params)

            dataset_dir = self.external_dir / dataset
            existing_files = (
                list(dataset_dir.glob(f"{dataset}_page_*.json"))
                if dataset_dir.exists()
                else []
            )

            return {
                "dataset": dataset,
                "total_pages": total_pages,
                "existing_files": len(existing_files),
                "dataset_dir": str(dataset_dir),
                "config": params,
            }
        except Exception as e:
            logger.error(f"Failed to get info for dataset {dataset}: {e}")
            return {}
