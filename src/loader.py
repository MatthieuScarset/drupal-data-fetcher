"""Load datasets from BigQuery."""

from typing import Optional, Dict, Any

import pandas as pd
from google.cloud import bigquery
from loguru import logger


class Loader:
    """Loads datasets from BigQuery."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def load_dataset(
        self, dataset_id: str, table_id: str, query: Optional[str] = None
    ) -> pd.DataFrame:
        """Load dataset from BigQuery table."""
        try:
            if query:
                # Use custom query
                df = pd.read_gbq(query, project_id=self.project_id)
            else:
                # Load entire table
                table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
                query = f"SELECT * FROM `{table_ref}`"
                df = pd.read_gbq(query, project_id=self.project_id)

            logger.info(f"Loaded {len(df)} records from {dataset_id}.{table_id}")
            return df

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}.{table_id}: {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query and return results as DataFrame."""
        try:
            df = pd.read_gbq(query, project_id=self.project_id)
            logger.info(f"Query executed successfully, returned {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def list_tables(self, dataset_id: str) -> list:
        """List all tables in a BigQuery dataset."""
        try:
            dataset_ref = self.client.dataset(dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            table_names = [table.table_id for table in tables]
            logger.info(f"Found {len(table_names)} tables in dataset {dataset_id}")
            return table_names
        except Exception as e:
            logger.error(f"Failed to list tables in dataset {dataset_id}: {e}")
            raise

    def get_table_schema(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """Get schema information for a BigQuery table."""
        try:
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)

            schema_info = {
                "table_id": table.table_id,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created,
                "modified": table.modified,
                "schema": [
                    {
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode,
                        "description": field.description,
                    }
                    for field in table.schema
                ],
            }

            logger.info(f"Retrieved schema for {dataset_id}.{table_id}")
            return schema_info

        except Exception as e:
            logger.error(f"Failed to get schema for {dataset_id}.{table_id}: {e}")
            raise
