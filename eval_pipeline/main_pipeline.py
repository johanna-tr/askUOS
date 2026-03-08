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


def main():
    """Complete pipeline orchestrator including tracing and ragas scoring"""

    if not settings.eval_mode_enabled:
        logger.info("[EVAL] Evaluation mode is not enabled...")
        return

    # Build folder name for evaluation run
    timestamp_baseline_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    run_name = f"{timestamp_baseline_start_str}_{eval_settings.run_name}/"

    # Switch to baseline model config
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
        save_path="baseline_trace",
        model_name=baseline.model_name,
        optional_model_name=baseline.optional_model_name,
        is_baseline=True,
    )
    # Initialize client for ragas evaluation
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OSS_BASE_URL,
    )
    EVALUATOR_LLM = llm_factory(EVALUATOR_LLM_NAME, client=client, max_tokens=100000)

    # Evaluate for baseline model config
    baseline_results_topic_adherence = call_topic_adherence(
        "baseline_trace",
        run_name,
        EVALUATOR_LLM,
    )
    save_to_csv_file(run_name, "baseline", baseline_results_topic_adherence)

    # Evaluate for baseline model config
    baseline_results_faithfulness = asyncio.run(
        eval_faithfulness(
            traces_path=f"{eval_settings.output_dir}{run_name}baseline_trace.csv",
            evaluator_llm=EVALUATOR_LLM,
        )
    )
    save_to_csv_file(run_name, "baseline", baseline_results_faithfulness)

    # Check model configs for evaluation
    if eval_settings.local_models == None:
        print("[EVAL] No local models for evaluation found.")
        return

    # Local models loop: Switch to local model config
    for i, model_cfg in enumerate(eval_settings.local_models, start=1):
        logger.info(
            f"[EVAL] --- Switching to Local Config {i}: {model_cfg.model_name}, {model_cfg.optional_model_name}---"
        )
        # Update model manager
        model_manager.set_active_models(
            model_name=model_cfg.model_name,
            optional_model_name=model_cfg.optional_model_name,
            local_config=model_cfg.is_model_local,
            optional_local_config=model_cfg.is_optional_model_local,
        )
        # Trace graph invokations
        local_trace_path = f"config_{i}_trace"
        running_trace_generator(
            run_name=run_name,
            save_path=local_trace_path,
            model_name=model_cfg.model_name,
            optional_model_name=model_cfg.optional_model_name,
            is_baseline=False,
        )

        # Check enabled metrics and return if no active metrics
        active_metrics = eval_settings.active_metrics
        if active_metrics == None:
            logger.info("[EVAL] No active metrics found.")
            return

        # Print config
        logger.info(f"[EVAL] Starting Evaluation Pipeline: {run_name}")
        logger.info(
            f"[EVAL] Models: {model_cfg.model_name}, {model_cfg.optional_model_name}"
        )
        logger.info(f"[EVAL] Context Window: {model_cfg.context_window}")
        active_metrics_str = ", ".join(active_metrics)
        logger.info(f"[EVAL] Metrics to be evaluated: {active_metrics_str}")

        # ---- Topic Adherence ----
        if "Topic Adherence" in active_metrics:
            # Evaluate for local model config
            results_topic_adherence = call_topic_adherence(
                f"config_{i}_trace",
                run_name,
                EVALUATOR_LLM,
            )
            save_to_csv_file(run_name, i, results_topic_adherence)

        # ---- Tool Call Accuracy ----
        if "Tool Call Accuracy" in active_metrics:
            results_tool_call_accuracy = asyncio.run(
                eval_tool_call_accuracy(
                    baseline_traces_path=f"{eval_settings.output_dir}{run_name}baseline_trace.csv",
                    eval_traces_path=f"{eval_settings.output_dir}{run_name}{local_trace_path}.csv",
                )
            )
            save_to_csv_file(run_name, i, results_tool_call_accuracy)

        # # ---- Tool Call F1 ----
        if "Tool Call F1" in active_metrics:
            results_tool_call_f1 = asyncio.run(
                eval_tool_call_f1(
                    baseline_traces_path=f"{eval_settings.output_dir}{run_name}baseline_trace.csv",
                    eval_traces_path=f"{eval_settings.output_dir}{run_name}{local_trace_path}.csv",
                )
            )
            save_to_csv_file(run_name, i, results_tool_call_f1)

        # # ---- Faithfulness ----
        if "Faithfulness" in active_metrics:
            # Evaluate for local model config
            results_faithfulness = asyncio.run(
                eval_faithfulness(
                    traces_path=f"{eval_settings.output_dir}{run_name}{local_trace_path}.csv",
                    evaluator_llm=EVALUATOR_LLM,
                )
            )
            save_to_csv_file(run_name, i, results_faithfulness)

    logger.info("✅ [EVAL] Evaluation done (main_pipeline.py).")
    return


def running_trace_generator(
    run_name: str,
    save_path: str,
    model_name: str,
    optional_model_name: str,
    is_baseline: bool,
):
    generator = TraceGenerator(
        run_name,
        save_path,
        model_name,
        optional_model_name,
        is_baseline,
    )
    # Handle the complete tracing process
    generator.load_queries()
    generator.initialize_agent()
    generator.generate_traces()
    generator.save_to_csv()


# Wrapper function for Topic Adherence metric
def call_topic_adherence(
    trace_path: str,
    run_name: str,
    EVALUATOR_LLM,
):

    REFERENCE_TOPICS = eval_settings.reference_topics
    MODES = eval_settings.topic_adherence_modes
    if MODES and REFERENCE_TOPICS:
        results_topic_adherence = asyncio.run(
            eval_topic_adherence(
                traces_path=f"{eval_settings.output_dir}{run_name}{trace_path}.csv",
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
    os.makedirs(output_dir, exist_ok=True)

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
