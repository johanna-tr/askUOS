import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics.collections import ToolCallAccuracy, ToolCallF1

from eval_pipeline.eval_helpers import load_traces


def get_valid_string(value):
    """Gives back string if valid string, otherwise ''."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def extract_samples(
    baseline_traces: List[Dict[str, Any]], eval_traces: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build samples for tool call related metrics."""
    samples: List[Dict] = []

    # Retrieve the relevant data columns from local trace
    # Conversation consists of HumanMessage and AIMessage (with tool calls)
    for eval_row in eval_traces:

        query_id = get_valid_string(eval_row.get("query_id"))

        chatbot_response_de = get_valid_string(eval_row.get("chatbot_response_de"))
        chatbot_response_en = get_valid_string(eval_row.get("chatbot_response_en"))

        # Build HumanMessage (optional)
        # user_initial_query_de = get_valid_string(eval_row.get("user_initial_query_de"))
        # user_initial_query_en = get_valid_string(eval_row.get("user_initial_query_en"))
        # human_de = HumanMessage(content=user_initial_query_de)
        # human_en = HumanMessage(content=user_initial_query_en)

        tc_name_de = get_valid_string(eval_row.get("last_tool_call_de"))
        tc_name_en = get_valid_string(eval_row.get("last_tool_call_en"))

        tc_args_de_str = get_valid_string(eval_row.get("tc_args_de"))
        tc_args_de = ast.literal_eval(tc_args_de_str) if tc_args_de_str.strip() else {}

        tc_args_en_str = get_valid_string(eval_row.get("tc_args_en"))
        tc_args_en = ast.literal_eval(tc_args_en_str) if tc_args_en_str.strip() else {}

        # Build tool calls
        tool_calls_de = [ToolCall(name=tc_name_de, args=tc_args_de)]
        tool_calls_en = [ToolCall(name=tc_name_en, args=tc_args_en)]

        # Build AIMessage
        ai_de = AIMessage(content=chatbot_response_de, tool_calls=tool_calls_de)
        ai_en = AIMessage(content=chatbot_response_en, tool_calls=tool_calls_en)

        turn_de = [ai_de]
        turn_en = [ai_en]

        # Fetch row in baseline_traces with same query_id
        df = pd.DataFrame(baseline_traces)
        matching = df[df["query_id"] == query_id]

        # If matching row was found, extract tool calls as reference
        if not matching.empty:
            reference_row = matching.iloc[0]

            ref_tc_args_de = ast.literal_eval(reference_row.get("tc_args_de"))
            ref_tc_args_en = ast.literal_eval(reference_row.get("tc_args_en"))

            reference_tc_de = [
                ToolCall(
                    name=get_valid_string(reference_row.get("last_tool_call_de")),
                    args=ref_tc_args_de,
                )
            ]
            reference_tc_en = [
                ToolCall(
                    name=get_valid_string(reference_row.get("last_tool_call_en")),
                    args=ref_tc_args_en,
                )
            ]
            # Append German sample
            samples.append(
                {
                    "query_id": query_id,
                    "language": "Deutsch",
                    "user_input": turn_de,
                    "reference_tool_calls": reference_tc_de,
                    "actual_tool_calls": tool_calls_de,
                }
            )
            # Append English sample
            samples.append(
                {
                    "query_id": query_id,
                    "language": "English",
                    "user_input": turn_en,
                    "reference_tool_calls": reference_tc_en,
                    "actual_tool_calls": tool_calls_en,
                }
            )

    return samples


async def eval_tool_call_accuracy(baseline_traces_path: str, eval_traces_path: str):
    """Evaluate Tool Call Accuracy"""
    # Load traces and build samples from them
    baseline_traces = load_traces(baseline_traces_path)
    eval_traces = load_traces(eval_traces_path)
    samples = extract_samples(baseline_traces, eval_traces)
    print(f"Loaded {len(samples)} samples")

    rows: List[Dict[str, Any]] = []  # for results

    metric = ToolCallAccuracy()  # Initialize ragas metric

    for sample in samples:
        query_id = sample["query_id"]
        try:
            print("[ToolCallAccuracy] === SAMPLE SENT TO EVAL ===: ", sample)

            # Calculate score
            result = await metric.ascore(
                user_input=sample["user_input"],
                reference_tool_calls=sample["reference_tool_calls"],
            )
            print(f"ToolCallAccuracy: {result.value}")

            # Append result
            rows.append(
                {
                    "metric": "ToolCallAccuracy",
                    "query_id": query_id,
                    "language": sample["language"],
                    "tool_call_accuracy_score": float(result.value),
                    "reference_tool_calls": json.dumps(
                        sample["reference_tool_calls"], default=str
                    ),
                    "actual_tool_calls": json.dumps(
                        sample["actual_tool_calls"], default=str
                    ),
                    "error_msg": "",
                }
            )

        except Exception as e:
            print(f"  [{len(samples)}] {query_id}: Error - {str(e)[:200]}")
            rows.append(
                {
                    "metric": "ToolCallAccuracy",
                    "query_id": query_id,
                    "language": sample["language"],
                    "tool_call_accuracy_score": "",
                    "reference_tool_calls": json.dumps(
                        sample["reference_tool_calls"], default=str
                    ),
                    "actual_tool_calls": json.dumps(
                        sample["actual_tool_calls"], default=str
                    ),
                    "error_msg": e,
                }
            )

    return {"metric": "ToolCallAccuracy", "rows": rows}


async def eval_tool_call_f1(baseline_traces_path: str, eval_traces_path: str):
    """Evaluate Tool Call F1"""
    # Load traces and build samples from them
    baseline_traces = load_traces(baseline_traces_path)
    eval_traces = load_traces(eval_traces_path)
    samples = extract_samples(baseline_traces, eval_traces)

    print(f"Loaded {len(samples)} samples")

    rows: List[Dict[str, Any]] = []  # for results

    metric = ToolCallF1()  # Initialize ragas metric

    for sample in samples:
        query_id = sample["query_id"]
        try:
            print("[ToolCallF1] === SAMPLE SENT TO EVAL ===: ", sample)

            # Calculate score
            result = await metric.ascore(
                user_input=sample["user_input"],
                reference_tool_calls=sample["reference_tool_calls"],
            )
            print(f"ToolCallF1: {result.value}")

            # Append result
            rows.append(
                {
                    "metric": "ToolCallF1",
                    "query_id": query_id,
                    "tool_call_f1_score": float(result.value),
                    "reference_tool_calls": json.dumps(
                        sample["reference_tool_calls"], default=str
                    ),
                    "actual_tool_calls": json.dumps(
                        sample["actual_tool_calls"], default=str
                    ),
                    "error_msg": "",
                }
            )

        except Exception as e:
            print(f"  [{len(samples)}] {query_id}: Error - {str(e)[:200]}")
            rows.append(
                {
                    "metric": "ToolCallF1",
                    "query_id": query_id,
                    "language": sample["language"],
                    "tool_call_f1_score": "",
                    "reference_tool_calls": json.dumps(
                        sample["reference_tool_calls"], default=str
                    ),
                    "actual_tool_calls": json.dumps(
                        sample["actual_tool_calls"], default=str
                    ),
                    "error_msg": e,
                }
            )

    return {"metric": "ToolCallF1", "rows": rows}
