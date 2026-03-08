import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.chatbot_log.chatbot_logger import logger
from src.config.core_config import settings

env_path = Path(__file__).parent.parent.parent.parent.parent / ".env.dev"
load_dotenv(env_path)


# TODO the cached AI answer should contained the sources of the information.
# TODO use vectordb to cache the AI answers
class CustomMemoryCache(InMemoryCache):
    def lookup(self, prompt: str, llm_string: str):
        """Look up based on prompt and llm_string.

        Args:
            prompt: a string representation of the prompt.
                In the case of a Chat model, the prompt is a non-trivial
                serialization of the prompt into the language model.
            llm_string: A string representation of the LLM configuration.

        Returns:
            On a cache miss, return None. On a cache hit, return the cached value.
        """
        result_lookup = self._cache.get((prompt, llm_string), None)
        if result_lookup:
            print(f"Cache hit for prompt: {prompt}")
        return result_lookup


# set_llm_cache(CustomMemoryCache())
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))


# --- NEW MODEL MANAGER ---
DEFAULT_MODEL = settings.model.model_name
DEFAULT_OPTIONAL_MODEL = settings.model.optional_model_name
OSS_BASE_URL = os.getenv("OSS_BASE_URL")
OPENAI_API_KEY = os.getenv("OSS_OPENAI_API_KEY")  # local model key


@dataclass(frozen=True)
class EndpointConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ModelManager:
    """
    Singleton that holds the CURRENT model configuration.
    Gets updates from main_pipeline.py via set_active_models().
    """

    _instance = None
    is_llm_local = False
    is_optional_llm_local = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)

            # Default: Standard production mode, takes models from global settings
            cls._instance.current_model_name = DEFAULT_MODEL
            cls._instance.current_optional_model_name = DEFAULT_OPTIONAL_MODEL

            # Default: No base_url override, key can come from .env.dev
            cls._instance.current_endpoint = EndpointConfig()
            cls._instance.current_optional_endpoint = EndpointConfig()

            cls._instance._llm_instance = None
            cls._instance._llm_optional_instance = None
        return cls._instance

    def set_active_models(
        self,
        model_name: str,
        optional_model_name: str,
        local_config: bool,
        optional_local_config: bool,
    ):
        """
        Switches the current models dynamically.
        Clears stored instances so they are re-created with new names.
        """
        logger.info(f"[EVAL] Switching models to: {model_name} / {optional_model_name}")
        self.current_model_name = model_name
        self.current_optional_model_name = optional_model_name

        local_endpoint = EndpointConfig(base_url=OSS_BASE_URL, api_key=OPENAI_API_KEY)

        # Check whether current models are local -> selecting local_endpoint or default EndpointConfig()
        if local_config is True:
            self.current_endpoint = local_endpoint
            self._is_llm_local = True
        else:
            self.current_endpoint = EndpointConfig()  # OpenAI default
            self._is_llm_local = False

        if optional_local_config is True:
            self.current_optional_endpoint = local_endpoint
            self._is_optional_llm_local = True
        else:
            self.current_optional_endpoint = EndpointConfig()  # OpenAI default
            self._is_optional_llm_local = False

        # Reset instances
        self._llm_instance = None
        self._llm_optional_instance = None

    def _build_llm(self, model_name: str, endpoint: EndpointConfig) -> ChatOpenAI:
        """Builds LLM ChatOpenAI instance"""
        logger.info(
            f"[EVAL] Building LLM: model={model_name}, base_url={'Yes' if endpoint.base_url else 'None'}, api_key={'Yes' if endpoint.api_key else 'None'}"
        )
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            streaming=True,
            callbacks=[StdOutCallbackHandler()],
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
        )

    def get_llm(self):
        """Returns LLM instance. Initiates building new instance if it does not exist."""
        if self._llm_instance is None:
            self._llm_instance = self._build_llm(
                self.current_model_name, self.current_endpoint
            )

        return self._llm_instance

    def get_llm_optional(self):
        """Returns secondary LLM instance. Initiates building new instance if it does not exist."""
        if self._llm_optional_instance is None:
            self._llm_optional_instance = self._build_llm(
                self.current_optional_model_name, self.current_optional_endpoint
            )
        return self._llm_optional_instance


# Global model manager instance
model_manager = ModelManager()


class DynamicChatLlm:
    def __call__(self, messages=None, *args, **kwargs):
        # Return self (proxy object) if called without args
        if messages is None:
            return self
        # Otherwise retrieve actual LLM and forward the call with all arguments
        instance = model_manager.get_llm()
        return instance(messages, *args, **kwargs)

    def __getattr__(self, name):
        # Forward attribute access (such as .invoke, .with_config) to LLM instance
        return getattr(model_manager.get_llm(), name)


class DynamicChatLlmOptional:
    def __call__(self, messages=None, *args, **kwargs):
        # Return self (proxy object) if called without args
        if messages is None:
            return self
        # Otherwise retrieve actual LLM and forward the call with all arguments
        instance = model_manager.get_llm_optional()
        return instance(messages, *args, **kwargs)

    def __getattr__(self, name):
        # Forward attribute access (such as .invoke, .with_config) to LLM instance
        return getattr(model_manager.get_llm_optional(), name)


llm = DynamicChatLlm()
llm_optional = DynamicChatLlmOptional()


# --- FALLBACK ---
# fallback method when json parsing fails with gpt-oss (used in grade_documents and judge_node)
def fallback_without_str_output(
    llm_model, prompt, invoke_kwargs: dict, required_field: str
) -> dict:
    """
    Fallback method for grading documents and judging agent decisions.
    without with_structured_output. Extracts JSON and binary from raw text.
    Measures and stores latency in fallback_latencies if enabled in global settings.
    """
    start_time = time.perf_counter()

    # Extract question and context from kwargs
    question = invoke_kwargs["question"]
    context = invoke_kwargs["context"]

    # Build chain and invoke LLM with streaming enabled
    llm = llm_model.with_config(streaming=True)
    chain = prompt | llm

    raw = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )
    # Extract and search for JSON in response
    content = raw.content.strip()
    json_match = re.search(r"(\{.*\})", content, re.DOTALL)
    json_str = json_match.group(1) if json_match else content

    try:
        # Parse JSON and record latency if eval mode enabled
        data = json.loads(json_str)

        if settings.eval_mode_enabled:
            latency = time.perf_counter() - start_time
            fallback_latencies[f"{required_field}_latency"] = latency

        return data  # as dict
    except (json.JSONDecodeError, ValidationError) as e:
        # Fallback: extract binary answer from text
        logger.error(f"[EVAL] Failed to parse json: {e}\nRaw content: {content}")
        binary = (
            "yes" if any(word in content.lower() for word in ["yes", "ja"]) else "no"
        )
        # Record latency
        if settings.eval_mode_enabled:
            latency = time.perf_counter() - start_time
            key = f"{required_field}_latency"
            # Add latency to previous latency if key already exists
            fallback_latencies[key] = fallback_latencies.get(key, 0) + latency
        return {required_field: binary, "reason": content[:300]}


# Dict for storing latency of fallback_without_str_output
fallback_latencies = {}


def reset_fallback_latencies():
    """Reset fallback_latencies dict."""
    global fallback_latencies
    fallback_latencies.clear()
    return


# --- previous LLM classes ---
# OPEN_AI_MODEL = settings.model.model_name

# class ChatLlm:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(ChatLlm, cls).__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if not self.__dict__:

#             self.llm_chat_open_ai = ChatOpenAI(
#                 model=OPEN_AI_MODEL,
#                 temperature=0,
#                 streaming=True,
#                 callbacks=[StdOutCallbackHandler()],
#             )

#     def __call__(self, *args, **kwargs) -> Any:
#         return self.llm_chat_open_ai


# class ChatLlmOptional:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(ChatLlmOptional, cls).__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if not self.__dict__:

#             self.llm_chat_open_ai = ChatOpenAI(
#                 model="gpt-4.1-nano",
#                 temperature=0,
#                 streaming=True,
#                 callbacks=[StdOutCallbackHandler()],
#             )

#     def __call__(self, *args, **kwargs) -> Any:
#         return self.llm_chat_open_ai


# llm = ChatLlm()
# # currenlty being used for summarization and grading documents (edge grade_documents (graph))
# llm_optional = ChatLlmOptional()
