import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from main_pipeline import running_trace_generator, save_to_csv_file
from metrics.faithfulness import eval_faithfulness
from metrics.tool_call_accuracy_and_f1 import eval_tool_call_accuracy, eval_tool_call_f1
from metrics.topic_adherence import eval_topic_adherence
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory

from eval_pipeline.config.eval_core_config import eval_settings
from eval_pipeline.trace_generator import TraceGenerator
from src.chatbot.agents.utils.agent_helpers import model_manager
from src.chatbot_log.chatbot_logger import logger
from src.config.core_config import settings

env_path = Path(__file__).parent.parent / ".env.dev"
load_dotenv(env_path)


def main():
    """
    Function for generating a single trace (using baseline config in eval_config).
    No metric evaulation.
    """

    if not settings.eval_mode_enabled:
        logger.info("[EVAL] Evaluation mode is not enabled...")
        return

    # Build folder name for evaluation run
    timestamp_baseline_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run_name = f"{timestamp_baseline_start_str}_{eval_settings.run_name}/"

    # Set model config
    baseline = eval_settings.baseline_model
    logger.info(
        f"[EVAL] --- Switching to baseline models: {baseline.model_name}, {baseline.optional_model_name}---"
    )
    model_manager.set_active_models(
        model_name=baseline.model_name,
        optional_model_name=baseline.optional_model_name,
        local_config=baseline.is_model_local,
        optional_local_config=baseline.is_optional_model_local,
    )
    # Trace graph invokations while using baseline model config
    running_trace_generator(
        run_name=run_name,
        save_path="trace",
        model_name=baseline.model_name,
        optional_model_name=baseline.optional_model_name,
        is_baseline=True,
    )

    logger.info("✅ [EVAL] Trace done (main_traces.py).")
    return


if __name__ == "__main__":
    main()
