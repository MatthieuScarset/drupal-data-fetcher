from loguru import logger
from typing import Optional, Dict
from urllib.parse import urljoin
import httpx
import time

from src.config import FETCHER_DEFAULT_HEADERS, FETCHER_BASE_URL

class Fetcher:
    """
    Fetcher for Drupal.org data using HTTPX with HTTP/2 support.
    Optimized for single-threaded, fast sequential requests.
    """

    def __init__(
        self,
        base_url: str = FETCHER_BASE_URL,
        cookies: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = 30,
        sleep: int = 10, # Retry-After header from d.o = 10sec
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.sleep = sleep
        self.client = httpx.Client(
            http2=True,
            timeout=timeout,
            headers={**FETCHER_DEFAULT_HEADERS, **(headers or {})},
            cookies=cookies or {}
        )
        self.logger = logger

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = 'GET',
        **kwargs
    ) -> Optional[httpx.Response]:
        """
        Make an HTTP request to the API.
        Returns a Response object or None if the request fails.
        """
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        try:
            response = self.client.request(
                method=method,
                url=url,
                params=params or {},
                **kwargs
            )
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            exc_type = type(e).__name__
            if status_code == 429:
                self.logger.warning(f"Rate limit exceeded: {url} | {exc_type}: {e}")
                self.logger.warning(f"Sleeping for {self.sleep} seconds")
                # Sleep a bit to avoid hitting the rate limit too quickly
                time.sleep(self.sleep)
            elif status_code == 503:
                self.logger.warning(f"Service unavailable: {url} | {exc_type}: {e}")
            else:
                self.logger.error(f"Request failed: {url} | {exc_type}: {e}")
            return None

    def get_total_pages(
        self,
        params: Optional[Dict] = None
    ) -> int:
        """
        Get the total number of pages for a resource.
        Returns total pages (default: 1 if not found).
        """
        request_params = (params or {}).copy()
        request_params['full'] = 0
        resource = request_params.pop('resource', 'node.json')
        response = self._make_request(resource, params=request_params)
        if response is None:
            raise Exception(f"No response")
        try:
            data = response.json()
            last_url = data.get('last')
            if not last_url or 'page=' not in last_url:
                return 0
            return int(last_url.split('page=')[1].split('&')[0]) + 1
        except Exception as e:
            self.logger.error(f"Error parsing total pages for {resource}: {e}")
            return 0

    def fetch_data(
        self,
        params: Optional[Dict] = None,
    ) -> Optional[httpx.Response]:
        """
        Fetch data from a specific resource.
        Returns a Response object or None if the request fails.
        """
        request_params = (params or {}).copy()
        endpoint = request_params.pop('resource')
        return self._make_request(endpoint, params=request_params)