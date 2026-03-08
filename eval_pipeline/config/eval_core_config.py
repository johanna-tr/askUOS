import sys
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chatbot_log.chatbot_logger import logger

from .eval_models import (
    ApplicationConfig,
    ChatPageConfig,
    EmbeddingSettings,
    Legal,
    LogSettings,
    MilvusSettings,
    ModelConfig,
    RAGFlowSettings,
    SearchConfig,
    StartPageConfig,
    VectorDBConfig,
)


class Eval_Settings(BaseSettings):
    """
    Settings class for evaluation configuration. Mostly adopted from global settings config.yaml.

    This class is a singleton that holds various configuration settings for the application.
    It inherits from `BaseSettings` and uses Pydantic for data validation and settings management.

    """

    _instance: ClassVar[Optional["Eval_Settings"]] = None

    # search_config: SearchConfig

    # ----- Start: This section specific to eval settings -----
    run_name: str
    output_dir: str
    input_dir: str
    eval_trace_path: str
    baseline_trace_path: str
    skip_baseline_scoring: bool
    recursion_limit: int
    baseline_model: ModelConfig
    local_models: List[ModelConfig] | None
    evaluator_llm: str
    active_metrics: List[str] | None
    topic_adherence_modes: List[str] | None
    reference_topics: List[str]
    # language: Literal["Deutsch", "English"] # language is disabled in eval_settings
    model_config = SettingsConfigDict(yaml_file="eval_pipeline/eval_config.yaml")
    # ----- End of section -----

    application: ApplicationConfig
    embedding: EmbeddingSettings
    vector_db_settings: VectorDBConfig
    start_page: StartPageConfig
    chat_page: ChatPageConfig
    legal: Optional[Legal] = (
        None  # Optional legal information (e.g., data protection, imprint)
    )
    # TODO move this a global object/context
    time_request_sent: Optional[float] = None
    # TODO remove (these are used for testing)
    # final_output_tokens: list = []
    # final_search_tokens: list = []
    # TODO move to a global object/context
    # if the llm summarization mode is active the summarization result will be not sent to the user
    llm_summarization_mode: bool = False
    log_settings: Optional[LogSettings] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Eval_Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self, **data):
        if not self.__dict__:
            super().__init__(**data)
            logger.debug(f"Settings initialized: {self.model_dump_json()}")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


eval_settings = Eval_Settings()
