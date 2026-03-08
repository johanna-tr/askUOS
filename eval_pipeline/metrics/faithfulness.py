import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from ragas.metrics.collections import Faithfulness

from eval_pipeline.eval_helpers import load_traces


def get_valid_string(value):
    """Gives back string if valid string, otherwise ''."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def extract_samples(
    traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build samples for faithfulness metric."""
    samples: List[Dict] = []

    for row in traces:

        query_id = get_valid_string(row.get("query_id"))
        # Retrieve the relevant data columns
        user_initial_query_de = get_valid_string(row.get("user_initial_query_de"))
        user_initial_query_en = get_valid_string(row.get("user_initial_query_en"))
        chatbot_response_de = get_valid_string(row.get("chatbot_response_de"))
        chatbot_response_en = get_valid_string(row.get("chatbot_response_en"))
        retrieved_context_de = get_valid_string(row.get("retrieved_context_de"))
        retrieved_context_en = get_valid_string(row.get("retrieved_context_en"))

        if user_initial_query_de != "" and chatbot_response_de != "":
            # Append German sample
            samples.append(
                {
                    "query_id": query_id,
                    "user_input": user_initial_query_de,
                    "response": chatbot_response_de,
                    "retrieved_contexts": [retrieved_context_de],
                    "language": "Deutsch",
                }
            )

        if user_initial_query_en != "" and chatbot_response_en != "":
            # Append English sample
            samples.append(
                {
                    "query_id": query_id,
                    "user_input": user_initial_query_en,
                    "response": chatbot_response_en,
                    "retrieved_contexts": [retrieved_context_en],
                    "language": "English",
                }
            )

    return samples


async def eval_faithfulness(traces_path: str, evaluator_llm):
    """Evaluate Faithfulness"""
    # Load traces and build samples from them
    traces = load_traces(traces_path)
    samples = extract_samples(traces)
    print(f"Loaded {len(samples)} samples")

    rows: List[Dict[str, Any]] = []  # for results

    metric = Faithfulness(llm=evaluator_llm)  # Initialize ragas metric

    for sample in samples:
        query_id = sample["query_id"]
        try:
            print(
                f"[FAITHFULNESS] === SAMPLE with query_id {query_id} SENT TO EVAL ===:",
                sample["user_input"],
            )
            await asyncio.sleep(3)

            # Calculate score
            result = await metric.ascore(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
            )
            print(f"Faithfulness: {result.value}")

            # Append result
            rows.append(
                {
                    "metric": "Faithfulness",
                    "query_id": query_id,
                    "language": sample["language"],
                    "faithfulness_score": float(result.value),
                    "error_msg": "",
                }
            )
        except Exception as e:
            print(f"  [{len(samples)}] {query_id}: Error - {str(e)[:200]}")
            # Append result
            rows.append(
                {
                    "metric": "Faithfulness",
                    "query_id": query_id,
                    "language": sample["language"],
                    "faithfulness_score": "",
                    "error_msg": e,
                }
            )

    return {"metric": "Faithfulness", "rows": rows}
