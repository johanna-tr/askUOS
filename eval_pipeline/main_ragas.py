import asyncio
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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

EVALUATOR_LLM_NAME = eval_settings.evaluator_llm
OSS_BASE_URL = os.getenv("OSS_BASE_URL")
OPENAI_API_KEY = os.getenv("OSS_OPENAI_API_KEY")

eval_trace_path = eval_settings.eval_trace_path
baseline_trace_path = eval_settings.baseline_trace_path


# Function for scoring the Ragas metrics
# Define active metrics in eval_config.yaml
# Set run_name in eval_config.yaml to the path of
def main():
    """Function for scoring Ragas metrics (no tracing)"""

    if not settings.eval_mode_enabled:
        logger.info("[EVAL] Evaluation mode is not enabled...")
        return

    # Build folder name for evaluation run
    timestamp_baseline_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run_name = f"{timestamp_baseline_start_str}_{eval_settings.run_name}/"

    # Construct saving path
    output_dir = os.path.join(eval_settings.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Print config
    logger.info(f"[EVAL] Starting Evaluation Pipeline: {run_name}")
    logger.info(f"[EVAL] Eval trace: {eval_trace_path}")
    logger.info(f"[EVAL] Baseline trace: {baseline_trace_path}")

    # Check enabled metrics and return if no active metrics
    active_metrics = eval_settings.active_metrics
    if active_metrics == None:
        logger.info("[EVAL] No active metrics found.")
        return

    active_metrics_str = ", ".join(active_metrics)
    logger.info(f"[EVAL] Metrics to be evaluated: {active_metrics_str}")

    # Initialize client for ragas evaluation
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OSS_BASE_URL,
    )
    EVALUATOR_LLM = llm_factory(
        EVALUATOR_LLM_NAME, client=client, max_tokens=100000
    )  # max_tokens for Faithfulness

    if not eval_settings.skip_baseline_scoring:
        # Evaluate for baseline model config
        baseline_results_topic_adherence = call_topic_adherence(
            baseline_trace_path,
            EVALUATOR_LLM,
        )
        save_to_csv_file(run_name, "baseline", baseline_results_topic_adherence)

        # Evaluate for baseline model config
        baseline_results_faithfulness = asyncio.run(
            eval_faithfulness(
                traces_path=f"{baseline_trace_path}.csv",
                evaluator_llm=EVALUATOR_LLM,
            )
        )
        save_to_csv_file(run_name, "baseline", baseline_results_faithfulness)

    # ---- Topic Adherence ----
    if "Topic Adherence" in active_metrics:
        # Evaluate for local model config
        results_topic_adherence = call_topic_adherence(
            eval_trace_path,
            EVALUATOR_LLM,
        )
        save_to_csv_file(run_name, "eval", results_topic_adherence)

    # ---- Tool Call Accuracy ----
    if "Tool Call Accuracy" in active_metrics:
        results_tool_call_accuracy = asyncio.run(
            eval_tool_call_accuracy(
                baseline_traces_path=f"{baseline_trace_path}.csv",
                eval_traces_path=f"{eval_trace_path}.csv",
            )
        )
        save_to_csv_file(run_name, "eval", results_tool_call_accuracy)

    # # ---- Tool Call F1 ----
    if "Tool Call F1" in active_metrics:
        results_tool_call_f1 = asyncio.run(
            eval_tool_call_f1(
                baseline_traces_path=f"{baseline_trace_path}.csv",
                eval_traces_path=f"{eval_trace_path}.csv",
            )
        )
        save_to_csv_file(run_name, "eval", results_tool_call_f1)

    # # ---- Faithfulness ----
    if "Faithfulness" in active_metrics:
        # Evaluate for local model config
        results_faithfulness = asyncio.run(
            eval_faithfulness(
                traces_path=f"{eval_trace_path}.csv",
                evaluator_llm=EVALUATOR_LLM,
            )
        )
        save_to_csv_file(run_name, "eval", results_faithfulness)

    logger.info("✅ [EVAL] Evaluation done (main_ragas.py).")
    return


# Wrapper function for Topic Adherence metric
def call_topic_adherence(
    trace_path: str,
    EVALUATOR_LLM,
):

    REFERENCE_TOPICS = eval_settings.reference_topics
    MODES = eval_settings.topic_adherence_modes
    if MODES and REFERENCE_TOPICS:
        results_topic_adherence = asyncio.run(
            eval_topic_adherence(
                traces_path=f"{trace_path}.csv",
                reference_topics=REFERENCE_TOPICS,
                evaluator_llm=EVALUATOR_LLM,
                modes=MODES,
            )
        )
        return results_topic_adherence
    else:
        logger.warning("[EVAL] Topic Adherence: No MODES or no REFERENCE TOPICS found.")
        return {}


def save_to_csv_file(run_name: str, description: int, results: dict):
    date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Extract eval results
    metric = results["metric"]
    data_to_write = results["rows"]

    # Construct saving path
    output_dir = eval_settings.output_dir
    output_file = os.path.join(
        output_dir, run_name, f"{date_str}_config_{description}_{metric}_results.csv"
    )

    # Save eval results to csv file
    if data_to_write:
        df = pd.DataFrame(data_to_write)
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            df.to_csv(f, index=False)

        print(f"[EVAL] Successfully saved {len(data_to_write)} rows to {output_file}")
    else:
        print("[EVAL] No data rows found to save.")
    return


if __name__ == "__main__":
    main()
