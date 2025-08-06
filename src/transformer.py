"""Transform JSON files to CSV and Parquet formats."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


class Transformer:
    """Transforms JSON data to CSV and Parquet formats."""

    def __init__(self):
        # Use config defaults
        self.external_dir = EXTERNAL_DATA_DIR
        self.interim_dir = INTERIM_DATA_DIR
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.data_dir = self.external_dir.parent

        # Create directories if they don't exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def merge_json_to_csv(self, entity_type: str, chunk_size: int = 1000) -> Path:
        """Merge JSON files for an entity type into a single CSV using chunked processing."""
        entity_dir = self.external_dir / entity_type
        if not entity_dir.exists():
            raise FileNotFoundError(f"Entity directory not found: {entity_dir}")

        json_files = list(entity_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {entity_dir}")

        csv_path = self.interim_dir / f"{entity_type}_merged.csv"
        total_records = 0
        first_chunk = True

        # Process files in chunks to manage memory
        for json_file in json_files:
            try:
                records = self._process_json_file_in_chunks(json_file, chunk_size)
                if records:
                    total_records += len(records)
                    # Write to CSV in append mode after first chunk
                    self._write_records_to_csv(records, csv_path, first_chunk)
                    first_chunk = False
                    logger.debug(f"Processed {json_file}: {len(records)} records")

            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                continue

        if total_records == 0:
            raise ValueError(f"No records found for entity type: {entity_type}")

        logger.info(
            f"Merged {len(json_files)} JSON files into {csv_path} ({total_records} records)"
        )
        return csv_path

    def _process_json_file_in_chunks(
        self, json_file: Path, chunk_size: Optional[int] = None
    ) -> List[dict]:
        """Process a single JSON file and return flattened records.

        Args:
            json_file: Path to the JSON file to process
            chunk_size: Currently unused, kept for future batch processing optimization

        Returns:
            List of flattened records
        """
        with open(json_file, "rb") as f:
            data = orjson.loads(f.read())

        # Handle different JSON structures
        records = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            if "list" in data:
                records = data["list"]
            else:
                records = [data]

        # Pre-process records to flatten nested objects
        processed_records = []
        for record in records:
            processed_record = self._flatten_nested_objects(record)
            processed_records.append(processed_record)

        # Note: chunk_size parameter reserved for future optimization
        # where we might process very large JSON files in batches
        return processed_records

    def _write_records_to_csv(
        self, records: List[dict], csv_path: Path, first_chunk: bool
    ):
        """Write records to CSV file in chunks."""
        if not records:
            return

        # Convert records to DataFrame
        df = pd.json_normalize(records)

        # Write to CSV
        if first_chunk:
            df.to_csv(csv_path, index=False, mode="w")
        else:
            df.to_csv(csv_path, index=False, mode="a", header=False)

    def convert_csv_to_parquet(self, csv_path: Path, chunk_size: int = 1000) -> Path:
        """Convert CSV file to Parquet format using chunked processing."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        parquet_path = self.raw_dir / f"{csv_path.stem}.parquet"

        # Process CSV in chunks to manage memory
        chunk_list = []
        total_rows = 0

        try:
            for chunk in pd.read_csv(
                csv_path, chunksize=chunk_size, low_memory=False, on_bad_lines="skip"
            ):
                chunk_list.append(chunk)
                total_rows += len(chunk)
                logger.debug(f"Processed chunk ({csv_path.stem}): {len(chunk)} rows")

                # Write chunks periodically to avoid memory buildup
                if len(chunk_list) >= 10:  # Write every 10 chunks
                    self._write_chunks_to_parquet(
                        chunk_list, parquet_path, total_rows == len(chunk)
                    )
                    chunk_list = []

            # Write remaining chunks
            if chunk_list:
                self._write_chunks_to_parquet(
                    chunk_list,
                    parquet_path,
                    total_rows == sum(len(chunk) for chunk in chunk_list),
                )

        except pd.errors.EmptyDataError:
            logger.warning(f"Empty CSV file: {csv_path}")
            # Create empty parquet file
            empty_df = pd.DataFrame()
            self._write_dataframe_to_parquet(empty_df, parquet_path)

        logger.info(f"Converted {csv_path} to {parquet_path} ({total_rows} rows)")
        return parquet_path

    def _write_chunks_to_parquet(
        self, chunk_list: List[pd.DataFrame], parquet_path: Path, is_first_write: bool
    ):
        """Write DataFrame chunks to Parquet file."""
        if not chunk_list:
            return

        # Concatenate chunks
        df = pd.concat(chunk_list, ignore_index=True)

        # Ensure consistent data types to avoid Parquet conversion errors
        df = self._normalize_dataframe_types(df)

        # Write to parquet
        if is_first_write:
            self._write_dataframe_to_parquet(df, parquet_path)
        else:
            # For append mode, we need to read existing data and combine
            if parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Normalize types again after concatenation
                combined_df = self._normalize_dataframe_types(combined_df)
                self._write_dataframe_to_parquet(combined_df, parquet_path)
            else:
                self._write_dataframe_to_parquet(df, parquet_path)

    def process_entity(self, entity_type: str, chunk_size: int = 1000) -> Path:
        """Full pipeline: JSON -> CSV -> Parquet for an entity type with chunked processing."""
        logger.info(f"Processing entity type: {entity_type}")

        # Step 1: Merge JSON to CSV
        csv_path = self.merge_json_to_csv(entity_type, chunk_size)

        # Step 2: Convert CSV to Parquet
        self.convert_csv_to_parquet(csv_path, chunk_size)

        # Step 3: Clean Parquet dataset.
        cleaned_parquet_path = self.clean_parquet_dataset(entity_type, chunk_size)

        return cleaned_parquet_path

    def process_all_entities(self, chunk_size: int = 1000) -> List[Path]:
        """Process all entity types found in external directory with chunked processing."""
        entity_dirs = [d for d in self.external_dir.iterdir() if d.is_dir()]
        parquet_files = []

        for entity_dir in entity_dirs:
            try:
                parquet_path = self.process_entity(entity_dir.name, chunk_size)
                parquet_files.append(parquet_path)
            except Exception as e:
                logger.error(f"Failed to process entity {entity_dir.name}: {e}")
                continue

        logger.info(f"Processed {len(parquet_files)} entity types")
        return parquet_files

    def sanitize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize column names to be BigQuery compatible."""
        # Replace dots and other invalid characters with underscores
        df.columns = df.columns.str.replace(".", "_", regex=False)
        df.columns = df.columns.str.replace("-", "_", regex=False)
        df.columns = df.columns.str.replace(" ", "_", regex=False)
        # Remove any other special characters
        df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
        # Ensure columns start with letter or underscore
        df.columns = df.columns.str.replace(r"^[0-9]", "_", regex=True)
        return df

    def _flatten_nested_objects(self, record):
        """Recursively flatten nested objects to extract their IDs or values."""
        if not isinstance(record, dict):
            return record

        flattened_record = {}

        for key, value in record.items():
            if isinstance(value, dict):
                # Check if this is a typical Drupal API object with id, uri, resource
                if "id" in value and "uri" in value and "resource" in value:
                    # Extract just the ID for these API reference objects
                    flattened_record[key] = value["id"]
                elif "id" in value:
                    # If there's an ID but not the full API structure, still extract the ID
                    flattened_record[key] = value["id"]
                elif "value" in value:
                    # Some objects might have a 'value' field instead
                    flattened_record[key] = value["value"]
                else:
                    # If it's a complex object without obvious ID/value, keep as is
                    # json_normalize will handle it
                    flattened_record[key] = value
            elif isinstance(value, list):
                # Handle arrays - check if they contain objects with IDs
                if len(value) > 0 and isinstance(value[0], dict):
                    if "id" in value[0]:
                        # Extract all IDs from the array as comma-separated string
                        ids = []
                        for item in value:
                            if isinstance(item, dict) and "id" in item:
                                ids.append(str(item["id"]))
                        flattened_record[key] = ",".join(ids) if ids else ""
                    elif "value" in value[0]:
                        # Extract all values from the array as comma-separated string
                        values = []
                        for item in value:
                            if isinstance(item, dict) and "value" in item:
                                values.append(str(item["value"]))
                        flattened_record[key] = ",".join(values) if values else ""
                    else:
                        # Complex array objects, convert to string representation for consistency
                        flattened_record[key] = str(value)
                else:
                    # Simple array or empty array, convert to string for consistency
                    if len(value) == 0:
                        flattened_record[key] = ""
                    else:
                        # Convert simple array to comma-separated string
                        flattened_record[key] = ",".join(str(item) for item in value)
            else:
                # Simple value, keep as is
                flattened_record[key] = value

        return flattened_record

    def clean_parquet_dataset(self, entity_type: str, chunk_size: int = 1000) -> Path:
        """Clean a parquet dataset by removing empty rows and unnecessary columns using chunked processing."""
        raw_parquet_path = self.raw_dir / f"{entity_type}_merged.parquet"
        processed_parquet_path = self.processed_dir / f"{entity_type}_processed.parquet"

        if not raw_parquet_path.exists():
            raise FileNotFoundError(f"Raw parquet file not found: {raw_parquet_path}")

        logger.info(f"Cleaning parquet dataset: {entity_type}")

        # Process parquet file in chunks to manage memory
        total_rows_processed = 0
        total_rows_kept = 0
        first_chunk = True

        try:
            # Read parquet file info to get total rows
            parquet_file = pd.read_parquet(raw_parquet_path)
            original_shape = parquet_file.shape
            logger.info(f"Original dataset shape: {original_shape}")

            # Process in chunks
            del parquet_file  # Free memory

            for chunk in self._read_parquet_in_chunks(raw_parquet_path, chunk_size):
                total_rows_processed += len(chunk)

                # Clean the chunk
                cleaned_chunk = self._clean_dataframe_chunk(chunk)

                if not cleaned_chunk.empty:
                    total_rows_kept += len(cleaned_chunk)
                    # Write cleaned chunk
                    self._write_cleaned_chunk_to_parquet(
                        cleaned_chunk, processed_parquet_path, first_chunk
                    )
                    first_chunk = False

                logger.debug(
                    f"Processed chunk ({entity_type}): {len(chunk)} -> {len(cleaned_chunk)} rows"
                )

        except Exception as e:
            logger.error(f"Error during chunked processing: {e}")
            # Fallback to non-chunked processing for smaller files
            return self._clean_parquet_dataset_fallback(entity_type)

        logger.info(
            f"Cleaned dataset: {total_rows_processed} -> {total_rows_kept} rows"
        )
        logger.info(f"Cleaned dataset saved to: {processed_parquet_path}")

        return processed_parquet_path

    def _read_parquet_in_chunks(self, parquet_path: Path, chunk_size: int):
        """Read parquet file in chunks."""
        # Read parquet file in chunks using pandas
        df = pd.read_parquet(parquet_path)

        # Yield chunks
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size].copy()

        del df  # Free memory

    def _clean_dataframe_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a single DataFrame chunk."""
        # Drop rows with all NaN values
        df = df.dropna(how="all")

        # Drop duplicate rows if any
        df = df.drop_duplicates()

        # Drop useless columns
        cols_to_drop = [
            # General nodes' columns from Drupal
            "has_new_content",
            "flag_flag_tracker_follow_user",
            "feed_nid",
            "feeds_item_url",
            "feeds_item_guid",
            "book_ancestors",
            "sticky",
            "promote",
            "status",
            "edit_url",
            "url",
            "language",
            "is_new",
            # Comment related columns
            "comment_count_new",
            "comments",
        ]

        # Drop columns that exist in the DataFrame
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)

        # Sanitize column names for BigQuery compatibility
        df = self.sanitize_column_names(df)

        return df

    def _write_cleaned_chunk_to_parquet(
        self, df: pd.DataFrame, parquet_path: Path, is_first_chunk: bool
    ):
        """Write cleaned DataFrame chunk to Parquet file."""
        if df.empty:
            return

        # Normalize data types before writing
        df = self._normalize_dataframe_types(df)

        if is_first_chunk:
            self._write_dataframe_to_parquet(df, parquet_path)
        else:
            # For append mode, read existing and combine
            if parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Normalize types again after concatenation
                combined_df = self._normalize_dataframe_types(combined_df)
                self._write_dataframe_to_parquet(combined_df, parquet_path)
            else:
                self._write_dataframe_to_parquet(df, parquet_path)

    def _normalize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame column types to prevent Parquet conversion errors."""
        df = df.copy()

        for col in df.columns:
            # Convert all columns to string first to handle mixed types
            # This prevents ArrowInvalid errors during Parquet conversion
            if df[col].dtype == "object":
                # For object columns, ensure all values are strings
                df[col] = df[col].astype(str)
                # Replace 'nan' string with empty string for consistency
                df[col] = df[col].replace("nan", "")
            elif df[col].dtype in ["int64", "float64"]:
                # For numeric columns, convert to string to maintain consistency
                # since we're dealing with mixed data types from JSON
                df[col] = df[col].astype(str)
                df[col] = df[col].replace("nan", "")
            elif df[col].dtype == "bool":
                # For boolean columns, convert to string representation
                df[col] = df[col].astype(str)
                # Convert True/False to consistent string format
                df[col] = df[col].replace({"True": "true", "False": "false"})
            else:
                # For any other data types, convert to string
                # This handles datetime, category, and other dtypes
                df[col] = df[col].astype(str)
                df[col] = df[col].replace("nan", "")

        return df

    def _write_dataframe_to_parquet(self, df: pd.DataFrame, parquet_path: Path):
        """Write DataFrame to Parquet with explicit string schema to prevent conversion errors."""
        # Handle empty DataFrame case
        if df.empty:
            if len(df.columns) == 0:
                # Completely empty DataFrame, create with minimal structure
                df = pd.DataFrame({"placeholder": []})

        # Create explicit schema with all columns as strings
        schema_fields = []
        for col in df.columns:
            schema_fields.append(pa.field(col, pa.string()))

        schema = pa.schema(schema_fields)

        # Convert DataFrame to Arrow table with explicit schema
        table = pa.Table.from_pandas(df, schema=schema)

        # Write to parquet
        pq.write_table(table, parquet_path)

    def _clean_parquet_dataset_fallback(self, entity_type: str) -> Path:
        """Fallback method for cleaning smaller parquet datasets without chunking."""
        raw_parquet_path = self.raw_dir / f"{entity_type}_merged.parquet"
        processed_parquet_path = self.processed_dir / f"{entity_type}_processed.parquet"

        logger.info(f"Using fallback cleaning method for: {entity_type}")

        # Read the raw parquet file
        df = pd.read_parquet(raw_parquet_path)
        logger.info(f"Original dataset shape: {df.shape}")

        # Clean the dataframe
        df = self._clean_dataframe_chunk(df)
        logger.info(f"After cleaning: {df.shape}")

        # Normalize data types before saving
        df = self._normalize_dataframe_types(df)

        # Save cleaned dataset to processed directory
        self._write_dataframe_to_parquet(df, processed_parquet_path)
        logger.info(f"Cleaned dataset saved to: {processed_parquet_path}")

        return processed_parquet_path

    def clean_all_parquet_datasets(self, chunk_size: int = 1000) -> List[Path]:
        """Clean all parquet datasets in the raw directory using chunked processing."""
        raw_parquet_files = list(self.raw_dir.glob("*_merged.parquet"))
        cleaned_files = []

        logger.info(f"Found {len(raw_parquet_files)} raw parquet files to clean")

        for parquet_file in raw_parquet_files:
            # Extract entity type from filename
            entity_type = parquet_file.stem.replace("_merged", "")

            try:
                cleaned_path = self.clean_parquet_dataset(entity_type, chunk_size)
                cleaned_files.append(cleaned_path)
                logger.info(f"✓ Cleaned {entity_type}")
            except Exception as e:
                logger.error(f"✗ Failed to clean {entity_type}: {e}")
                continue

        logger.info(f"Successfully cleaned {len(cleaned_files)} datasets")
        return cleaned_files
