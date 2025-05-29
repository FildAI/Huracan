import os
import json
import requests
from typing import List, Dict, Any, Optional

class QdrantHttpClient:
    """
    Minimal HTTP-only Qdrant client (no qdrant-client dependency).
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        api_key: str = None,
        use_bearer: bool = False,
        timeout: int = 60,
    ):
        # Allow environment defaults
        self.host = host or os.getenv("QDRANT_URL", "http://localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", 6333))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.use_bearer = use_bearer
        self.timeout = timeout

    def _base_url(self) -> str:
        if not self.host.startswith(("http://", "https://")):
            self.host = f"http://{self.host}"
        return f"{self.host}:{self.port}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.use_bearer:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["api-key"] = self.api_key
        return headers

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        vector_name: str = "vector",
        timeout: Optional[int] = None,
    ) -> None:
        """
        Create a new collection with the given vector schema.
        """
        url = f"{self._base_url()}/collections/{collection_name}"
        body = {
            "vectors": {
                vector_name: {"size": vector_size, "distance": distance}
            }
        }
        params = {}
        if timeout is not None:
            params["timeout"] = timeout
        resp = requests.put(
            url, headers=self._headers(), json=body, params=params, timeout=self.timeout
        )
        resp.raise_for_status()

    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        wait: bool = False,
        ordering: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upsert (insert/update) a batch of points.
        Points must be list of {"id": ..., "vector": [...], "payload": {...}}  [oai_citation:16‡Qdrant](https://qdrant.tech/documentation/concepts/points/?utm_source=chatgpt.com).
        """
        url = f"{self._base_url()}/collections/{collection_name}/points"
        params = {"wait": str(wait).lower()}
        if ordering:
            params["ordering"] = ordering
        resp = requests.put(
            url, headers=self._headers(), json={"points": points},
            params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve_points(
        self, collection_name: str, ids: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Fetch points by ID. Returns list of point objects.  [oai_citation:17‡Qdrant API Reference](https://api.qdrant.tech/api-reference/points/get-points?utm_source=chatgpt.com)
        """
        url = f"{self._base_url()}/collections/{collection_name}/points"
        resp = requests.post(
            url, headers=self._headers(), json={"ids": ids}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json().get("result", [])

    def search_points(
        self,
        collection_name: str,
        vector: List[float],
        limit: int = 10,
        vector_name: str = "vector",
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a semantic search in the collection.  [oai_citation:18‡Pipedream](https://pipedream.com/integrations/upsert-point-with-qdrant-api-on-daily-schedule-from-schedule-api-int_dasN5mqp?utm_source=chatgpt.com)
        """
        url = f"{self._base_url()}/collections/{collection_name}/points/search"
        body: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "vector_name": vector_name,
        }
        if filter is not None:
            body["filter"] = filter
        resp = requests.post(
            url, headers=self._headers(), json=body, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json().get("result", {})

    def delete_points(
        self,
        collection_name: str,
        ids: Optional[List[Any]] = None,
        filter: Optional[Dict[str, Any]] = None,
        wait: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete points by IDs or by filter.  [oai_citation:19‡Pipedream](https://pipedream.com/integrations/upsert-point-with-qdrant-api-on-daily-schedule-from-schedule-api-int_dasN5mqp?utm_source=chatgpt.com)
        """
        url = f"{self._base_url()}/collections/{collection_name}/points/delete"
        body: Dict[str, Any] = {}
        if ids is not None:
            body["ids"] = ids
        if filter is not None:
            body["filter"] = filter
        resp = requests.post(
            url, headers=self._headers(), json=body,
            params={"wait": str(wait).lower()}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()