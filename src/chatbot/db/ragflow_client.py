import os
import threading
from typing import Any, List, NamedTuple, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from src.chatbot_log.chatbot_logger import logger
from src.config.core_config import settings

# TODO delete once metadata is added to RAGFlow API
FAQ_BASE_URL = "https://uni-osnabrueck.de/"
NUMBER_CHUNKS_RETRIEVE = 10  # number of chunks to retrieve


class RetrievedDocs(BaseModel):
    count: int
    doc_id: str
    doc_name: str


class Chunk(BaseModel):
    content: str
    content_ltks: str
    dataset_id: str
    doc_type_kwd: str
    document_id: str
    document_keyword: str  # documents name
    highlight: str
    id: str
    image_id: str
    important_keywords: List[str]
    positions: List[List[Any]]
    similarity: float
    term_similarity: float
    vector_similarity: float

    # TODO delete once metadata is added to RAGFlow API
    @property
    def url_reference_askuos(self) -> str:
        """Generate URL reference."""
        file_name = os.path.splitext(self.document_keyword.replace("_", "/"))[0]
        return f"{FAQ_BASE_URL}{file_name}"

    @property
    def page(self) -> int:
        """Compute the page number from positions."""
        if self.positions:
            return self.positions[0][0]
        return 0  # Default to 0 if positions are empty


class RetrievedContentRagflow(BaseModel):
    chunk: Chunk
    doc_aggs: RetrievedDocs


class RAGFlowSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RAGFlowSingleton, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("RAGFLOW_API_KEY")
        self.base_url = base_url or settings.vector_db_settings.settings.base_url
        self.dbs = {}
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def __del__(self) -> None:
        if self._session:
            self._session.close()

    def get_db_id(self, db_name: str) -> str:
        """Get the database ID for a given database name."""
        if db_name in self.dbs.keys():
            return self.dbs[db_name]

        try:
            resp = self._session.get(
                f"{self.base_url}/api/v1/datasets", params={"name": db_name}
            )
            if resp.status_code == 200:
                datasets = resp.json()
                if datasets and "data" in datasets and datasets["data"]:
                    self.dbs[db_name] = datasets["data"][0]["id"]
                    logger.debug(f"Database ID for '{db_name}': {self.dbs[db_name]}")
                    return datasets["data"][0]["id"]
                else:
                    logger.error(f"Database '{db_name}' not found.")
                    raise ValueError(f"Database '{db_name}' not found.")
            else:
                logger.error(f"Failed to fetch database ID: {resp.status_code}")
                raise ValueError(
                    f"Failed to fetch database ID: {resp.status_code} - {resp.text}"
                )
        except requests.RequestException as e:
            logger.error(f"Network error while fetching database ID: {e}")
            raise ValueError(f"Network error: {e}")

    # use async method to retrieve chunks aiohttp
    def retrieve_chunks(
        self,
        query: str,
        db_id: str,
        page_size: int = NUMBER_CHUNKS_RETRIEVE,  # number of chunks to retrieve
    ) -> List[Chunk]:
        """Retrieve chunks from the RAGFlow database based on a query."""
        try:
            resp = self._session.post(
                f"{self.base_url}/api/v1/retrieval",
                headers={"Content-Type": "application/json"},
                json={
                    "question": query,
                    "dataset_ids": [db_id],
                    "document_ids": [],
                    "page_size": page_size,
                    "cross_languages": ["German", "English"],
                },
            )
            if resp.status_code == 200:
                r = resp.json()
                retrieved_content = [
                    RetrievedContentRagflow(
                        chunk=Chunk(**chunk), doc_aggs=RetrievedDocs(**docs)
                    )
                    for chunk, docs in zip(r["data"]["chunks"], r["data"]["doc_aggs"])
                ]
                return retrieved_content
            else:
                logger.error(
                    f"[RAGFlow]Failed to retrieve content from RAGFlow: {resp.status_code}"
                )
                raise ValueError(
                    f"[RAGFlow]Failed to retrieve content from RAGFlow: {resp.status_code} - {resp.text}"
                )
        except requests.RequestException as e:
            logger.error(f"[RAGFlow]Network error while retrieving chunks: {e}")
            raise ValueError(f"Network error: {e}")

    def get_chunks(self, query: str, db_name: str) -> List[Chunk]:

        chunks = []
        try:
            db_id = self.get_db_id(db_name)
            chunks = self.retrieve_chunks(query, db_id)
            logger.debug(
                f"[RAGFlow]Retrieved {len(chunks)} chunks for query '{query}' in database '{db_name}'."
            )
        except Exception as e:
            logger.error(f"[RAGFlow]Error retrieving chunks: {e}")
        return chunks

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_chunks(*args, **kwds)


if __name__ == "__main__":
    # for testing purposes
    ragflowretriever = RAGFlowSingleton()

    r1 = ragflowretriever("Credit points computer science", "UOS_PO")
    r2 = ragflowretriever("Credit points biology", "UOS_PO")
    r3 = ragflowretriever("Leistungspunkte Informatik", "UOS_PO")

    print(r1, r2, r3)
