# Drupal Data Fetcher

A comprehensive data pipeline for extracting, transforming, and loading Drupal.org data into BigQuery.

## Project Structure

### Core Components

**extracter.py**
* Fetch JSON files from Drupal.org API

**transformer.py**
* Merge JSON files to CSV
* Convert CSV to Parquet RAW format

**cloud.py**
* Push files to GCP bucket
* Pull files from GCP bucket
* Insert Parquet to BigQuery

**loader.py**
* Load datasets from BigQuery
* Execute custom queries

**main.py**
* CLI application with commands for each pipeline step

**notebook.py**
* Display results and analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### CLI Commands

```bash
# Transform specific entity type
python main.py transform project

# Transform all entities
python main.py transform-all

# Upload to GCP bucket
python main.py upload data/raw/project_merged.parquet --bucket-name your-bucket --project-id your-project

# Load to BigQuery
python main.py load-to-bq data/raw/project_merged.parquet --project-id your-project --dataset-id drupal_data --table-id projects --bucket-name your-bucket

# Run full pipeline
python main.py pipeline project --project-id your-project --bucket-name your-bucket
```

### Environment Variables

Create `.env` file:
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GCP_PROJECT_ID=your-project-id
GCP_BUCKET_NAME=your-bucket-name
BIGQUERY_DATASET_ID=drupal_data
```