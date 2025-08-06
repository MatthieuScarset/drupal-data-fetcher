from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

DEV_MODE=False
if DEV_MODE:
    logger.info("Running in development mode")

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
COMPILED_DATA_DIR = DATA_DIR / "compiled"
INTERIM_DATA_DIR = DATA_DIR / "interim"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Fetcher configuration for Drupal.org
FETCHER_BASE_URL = "https://www.drupal.org/api-d7"
FETCHER_UPDATES_URL = "https://updates.drupal.org/release-history"
FETCHER_DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "DrupalDataFetcher/1.0"
}

# Dataset configuration.
DATASET_MAPPINGS = {
    # Projects
    'project': {
        'resource': 'node.json',
        'type[]': [
            # Contrib modules.
            # @see https://www.drupal.org/project/project_module
            'project_module',
            # Core projects.
            # @see https://www.drupal.org/project/project_core
            'project_core',
            # Community projects.
            # @see https://www.drupal.org/project/project_drupalorg
            'project_drupalorg',
            # Distributions.
            # @see https://www.drupal.org/project/project_distribution
            'project_distribution',
        ],
        'sort': 'created',
        'direction': 'ASC',
    },
    # Issues
    # @see https://www.drupal.org/project/issues
    'issue': {
        'resource': 'node.json',
        'type': 'project_issue',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Change notices
    # @see https://www.drupal.org/list-changes/drupal
    'changenotice': {
        'resource': 'node.json',
        'type': 'changenotice',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Core releases
    'release': {
        'resource': 'node.json',
        'type': 'project_release',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Forum posts
    # @see https://www.drupal.org/forum
    'forum': {
        'resource': 'node.json',
        'type': 'forum',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Organizations 
    # @see https://www.drupal.org/drupal-services
    'organization': {
        'resource': 'node.json',
        'type': 'organization',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Case studies
    # @see https://www.drupal.org/case-studies
    'casestudy': {
        'resource': 'node.json',
        'type': 'casestudy',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Users
    'user': {
        'resource': 'user.json',
        'sort': 'uid',
        'direction': 'ASC',
    },
    # Events
    # @see https://www.drupal.org/community/events
    'event': {
        'resource': 'node.json',
        'type': 'event',
        'sort': 'created',
        'direction': 'ASC',
    },
    # Taxonomy terms.
    # @see https://www.drupal.org/api-d7/taxonomy_term.json
    'term': {
        'resource': 'taxonomy_term.json',
        'sort': 'tid',
        'direction': 'ASC',
    },
    # Taxonomy vocabularies.
    # @see https://www.drupal.org/api-d7/taxonomy_vocabulary.json
    'vocabulary': {
        'resource': 'taxonomy_vocabulary.json',
        'sort': 'vid',
        'direction': 'ASC',
    },
    # [DISABLED] Comments
    # There is way too much data in comments, so we disable it by default.
    # More than 10.5 million comments as of 2025-06-01.
    # @see https://www.drupal.org/api-d7/comment.json?full=0&limit=1
    #'comment': {
    #    'resource': 'comment.json',
    #    'sort': 'created',
    #    'direction': 'ASC',
    #},
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
