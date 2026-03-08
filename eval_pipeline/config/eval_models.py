from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel

EmbeddingType = Literal["FastEmbed", "Ollama"]


class VectorDBTypes(str, Enum):
    MILVUS = "Milvus"
    INFINITY_RAGFLOW = "Infinity-RAGFlow"


class CollectionNames(str, Enum):
    """
    Collection names used in the application.
    """

    EXAMINATION_REGULATIONS = "examination_regulations"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"


class SearchConfig(BaseModel):
    """
    Configuration for the search service.
    """

    search_url: str
    service: str


class ModelConfig(BaseModel):
    """
    Configuration for the model being used. Eval Pipeline specific.
    """

    model_name: str
    is_model_local: bool
    optional_model_name: str
    is_optional_model_local: bool
    context_window: int  # Number of allowed tokens


class Legal(BaseModel):
    """
    Configuration for legal information.
    """

    data_protection: str
    imprint: str


class ApplicationConfig(BaseModel):
    """
    Configuration for the application.
    """

    debug: bool
    recursion_limit: int = 12
    tracing: bool = False
    opik_project_name: str = "askUOSTesting"


class EmbeddingConnectionSettings(BaseModel):
    """Settings for Ollama embeddings"""

    model_name: str = (
        "intfloat/multilingual-e5-large"  # e.g., "llama2", "mistral", "intfloat/multilingual-e5-large"
    )
    base_url: Optional[str] = "http://localhost:11434"
    headers: Optional[dict] = None


class EmbeddingSettings(BaseModel):
    """Settings for text embedding"""

    type: EmbeddingType = "FastEmbed"
    connection_settings: EmbeddingConnectionSettings
    chunk_size: int = (
        1800  # Size of each chunk in characters (Only crawler uses a text splitter)
    )
    chunk_overlap: int = 50
    batch_size: int = 256


class LogSettings(BaseModel):
    delete_logs_days: int = 90  # Number of days to keep logs before deletion


class MilvusSettings(BaseModel):
    """Settings for Milvus vector database"""

    uri: Optional[str] = "http://localhost:19530"
    host: Optional[str] = None
    port: int = 19530
    token: Optional[str] = "root:Milvus"


class RAGFlowSettings(BaseModel):
    """
    Configuration for RAGFlow settings.
    """

    base_url: str
    chunk_size: int = 10  # Number of chunks to retrieve per request


class VectorDBConfig(BaseModel):
    """
    Configuration for the vector database.
    """

    type: VectorDBTypes = VectorDBTypes.MILVUS
    settings: Union[MilvusSettings, RAGFlowSettings]


class StartPageConfig(BaseModel):
    """
    Configuration for the start page.
    """

    welcome_message_english: str
    welcome_message_german: str


class ChatPageConfig(BaseModel):
    """
    Configuration for the chat page.
    """

    delete_message_dialog_box_english: str
    delete_message_dialog_box_german: str
