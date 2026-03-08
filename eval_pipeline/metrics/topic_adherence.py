import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from ragas.metrics.collections import TopicAdherence

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_pipeline.eval_helpers import load_traces


def get_valid_string(value):
    """Gives back string if valid string, otherwise ''."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def extract_samples(
    traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build samples for topic adherence metric."""
    samples: List[Dict] = []

    for row in traces:

        query_id = get_valid_string(row.get("query_id"))
        # Retrieve the relevant data columns
        user_initial_query_de = get_valid_string(row.get("user_initial_query_de"))
        user_initial_query_en = get_valid_string(row.get("user_initial_query_en"))
        chatbot_response_de = get_valid_string(row.get("chatbot_response_de"))
        chatbot_response_en = get_valid_string(row.get("chatbot_response_en"))

        # Build conversation (HumanMessage and AIMessage)
        human_de = HumanMessage(content=user_initial_query_de)
        human_en = HumanMessage(content=user_initial_query_en)
        ai_de = AIMessage(content=chatbot_response_de, tool_calls=[])
        ai_en = AIMessage(content=chatbot_response_en, tool_calls=[])

        turn_de = [human_de, ai_de]
        turn_en = [human_en, ai_en]

        # Filtering
        # Append German sample if query and response exist
        if user_initial_query_de != "" and chatbot_response_de != "":
            samples.append(
                {"query_id": query_id, "user_input": turn_de, "language": "Deutsch"}
            )

        # Append English sample if query and response exist
        if user_initial_query_en != "" and chatbot_response_en != "":
            samples.append(
                {"query_id": query_id, "user_input": turn_en, "language": "English"}
            )

    return samples


async def eval_topic_adherence(
    traces_path: str, reference_topics: List[str], evaluator_llm, modes: List[str]
):
    """Evaluate Topic Adherence"""
    # Load traces and build samples from them
    traces = load_traces(traces_path)
    samples = extract_samples(traces)
    print(f"Loaded {len(samples)} samples.")

    rows: List[Dict[str, Any]] = []  # for results

    for mode in modes:
        metric = TopicAdherence(  # Initialize ragas metric for each mode, defined in eval_settings
            llm=evaluator_llm,
            mode=mode,  # can be precision, recall and f1
        )

        for sample in samples:
            query_id = sample["query_id"]

            try:
                print(
                    "[TopicAdherence] === SAMPLE SENT TO EVAL ===: ",
                    sample,
                    reference_topics,
                )
                await asyncio.sleep(3)
                # Calculate score
                result = await metric.ascore(
                    user_input=sample["user_input"], reference_topics=reference_topics
                )
                print(f"Topic Adherence ({mode}): {result.value}")

                # Append result
                rows.append(
                    {
                        "metric": "TopicAdherence",
                        "mode": mode,
                        "query_id": query_id,
                        "language": sample["language"],
                        "topic_adherence_score": float(result.value),
                        "error_msg": "",
                    }
                )
            except Exception as e:
                print(f"  [{len(samples)}] {query_id}: Error - {str(e)[:200]}")
                # Append result
                rows.append(
                    {
                        "metric": "TopicAdherence",
                        "mode": mode,
                        "query_id": query_id,
                        "language": sample["language"],
                        "topic_adherence_score": "",
                        "error_msg": e,
                    }
                )

    return {"metric": "TopicAdherence", "rows": rows}
