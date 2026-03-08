import csv
import json
import os
import time
import uuid

import pandas as pd
from eval_helpers import EvalTracer
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage

from eval_pipeline.config.eval_core_config import eval_settings
from src.chatbot.agents.agent_lang_graph import CampusManagementOpenAIToolsAgent
from src.chatbot.agents.utils.agent_helpers import (
    fallback_latencies,
    reset_fallback_latencies,
)
from src.chatbot.prompt.main import get_system_prompt
from src.chatbot.prompt.prompt_date import get_current_date
from src.chatbot_log.chatbot_logger import logger
from src.config.core_config import settings


class TraceGenerator:
    def __init__(
        self, run_name, save_path, model_name, optional_model_name, is_baseline
    ):
        self.query_data = []
        self.agent_de = None
        self.agent_en = None
        self.traces = []
        self.model_name: str = model_name
        self.optional_model_name: str = optional_model_name
        self.is_baseline: bool = is_baseline
        self.save_path: str = save_path

        # Create output directory for eval run
        self.output_dir = f"{eval_settings.output_dir}{run_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_queries(self):
        """Load German and English query data from csv file."""
        query_path = eval_settings.input_dir

        if not os.path.exists(query_path):
            raise FileNotFoundError(f"[EVAL] Query file not found: {query_path}")

        query_data = []

        # Read and save query data
        with open(query_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                query_id = row.get("query_id", "").strip()
                question_de = row.get("user_initial_query_de", "").strip()
                question_en = row.get("user_initial_query_en", "").strip()
                if query_id:
                    query_data.append(
                        {
                            "query_id": query_id,
                            "user_initial_query_de": question_de,
                            "user_initial_query_en": question_en,
                        }
                    )
        self.query_data = query_data
        return

    def initialize_agent(self):
        """Initialize chatbot agent for each language with an EvalTracer instance."""
        try:
            # German agent
            settings.language = "Deutsch"
            self.agent_de = CampusManagementOpenAIToolsAgent.run(language="Deutsch")
            eval_tracer_de = EvalTracer()
            self.agent_de._eval_tracer = eval_tracer_de

            # English agent
            settings.language = "English"
            self.agent_en = CampusManagementOpenAIToolsAgent.run(language="English")
            eval_tracer_en = EvalTracer()
            self.agent_en._eval_tracer = eval_tracer_en

            if id(self.agent_de) == id(self.agent_en):
                logger.error("[EVAL] Both agents share the same id.")

            logger.info("[EVAL] Agents initialized successfully")
        except Exception as e:
            logger.error(f"[EVAL] Failed to initialize agents: {e}")
            raise

    def generate_traces(self):
        """
        Run queries through the agents (selecting agent via language)
        Generate evaluation traces with EvalTracer instances and external fallback_latency.
        Create trace entries in self.traces.
        """
        if not self.agent_de or not self.agent_en:
            raise RuntimeError("[EVAL] Agents not initialized.")

        if not self.query_data:
            raise RuntimeError("[EVAL] FAQ data not loaded.")

        logger.info("[EVAL] ------- Query processing -------")
        for idx, query_item in enumerate(self.query_data, 1):
            try:
                # Initialize tracing dicts
                tracer_dict_de = None
                tracer_dict_en = None

                # Decompose query_item
                query_id: str = query_item.get("query_id", "")
                user_initial_query_de: str = query_item.get("user_initial_query_de", "")
                user_initial_query_en: str = query_item.get("user_initial_query_en", "")

                # Reset tracing dicts
                self.agent_de._eval_tracer.reset()
                reset_fallback_latencies()

                # Invoke graph with German query
                if query_id and user_initial_query_de:
                    tracer_dict_de = self.invoke_graph("Deutsch", user_initial_query_de)

                # Reset tracing dicts
                self.agent_en._eval_tracer.reset()
                reset_fallback_latencies()

                # Invoke graph with English query
                if query_id and user_initial_query_en:
                    tracer_dict_en = self.invoke_graph("English", user_initial_query_en)

                # Store traces for query_item (both English and German)
                self.create_trace_entry(
                    query_item,
                    tracer_dict_de,
                    tracer_dict_en,
                )

            except Exception as e:
                logger.error(f"Error querying agent for query {query_id}: {e}")
                raise

    def invoke_graph(self, language, user_initial_query):
        """Invoke graph and handle tracing data."""
        start_time_str = ""
        end_time_str = ""
        latency_total = 0.0

        try:
            # Create config for invokation
            settings.language = language
            thread_id = uuid.uuid4()
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": eval_settings.recursion_limit,
            }

            # Select agent based on language
            if language == "Deutsch":
                agent = self.agent_de
            elif language == "English":
                agent = self.agent_en
            else:
                print("This language is not supported")
                return

            # Create prompt based on query and language
            prompt = get_system_prompt(
                conversation_summary="",
                messages=[HumanMessage(content=user_initial_query)],
                user_input=user_initial_query,
                current_date=get_current_date(language),
            )
            logger.debug(f"[TRACE GENERATOR] prompt: {prompt}")

            # Invoke graph while measuring response time
            logger.info(f"[EVAL] Invoking agent for: {user_initial_query}")
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            start_time_latency = time.perf_counter()
            response = agent._graph.invoke(
                {
                    "messages": prompt,
                    "message_history": [HumanMessage(content=user_initial_query)],
                    "user_initial_query": user_initial_query,
                },
                config=config,
            )
            end_time_latency = time.perf_counter()
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            latency_total = end_time_latency - start_time_latency

            # Get dict from EvalTracer instance
            tracer_dict = agent._eval_tracer.get_attr_dict()

            # Add chatbot response and latency data
            messages = response.get("messages", [])
            if messages:
                tracer_dict["chatbot_response"] = messages[-1].content  # Last message
            tracer_dict["latency_total"] = latency_total
            tracer_dict["start_time"] = start_time_str
            tracer_dict["end_time"] = end_time_str

            print("[EVAL] tracer_dict keys:", list(tracer_dict.keys()))
            print("[EVAL] fallback_latencies keys: ", list(fallback_latencies.keys()))

            # merge both dicts
            final_tracer_dict = tracer_dict | fallback_latencies

            # print("[EVAL] final_tracer_dict keys:", list(final_tracer_dict.keys()))

            print("Invokation successful")
            return final_tracer_dict

        except GraphRecursionError as e:
            logger.error(f"[EVAL] Recursion limit reached: {e}")
            if hasattr(agent, "_eval_tracer"):
                tracer_dict = agent._eval_tracer.get_attr_dict()
                tracer_dict["start_time"] = start_time_str
                tracer_dict["graph_error_msg"] = (
                    "Recursion limit reached without hitting a stop condition."
                )
                final_tracer_dict = tracer_dict | fallback_latencies

            return final_tracer_dict

        except Exception as e:
            logger.error(
                f"[EVAL] Error invoking graph for query {user_initial_query}: {e}"
            )
            if hasattr(agent, "_eval_tracer"):
                tracer_dict = agent._eval_tracer.get_attr_dict()
                tracer_dict["start_time"] = start_time_str
                tracer_dict["graph_error_msg"] = e
                final_tracer_dict = tracer_dict | fallback_latencies

            return final_tracer_dict

    def create_trace_entry(
        self,
        faq_item,
        tracer_dict_de,
        tracer_dict_en,
    ):
        """Create a trace entry."""

        # Initialize defaults
        chatbot_response_de = ""
        chatbot_response_en = ""
        retrieved_context_de = ""
        retrieved_context_en = ""
        tool_name_de = ""
        tool_name_en = ""
        tc_id_de = ""
        tc_id_en = ""
        tc_search_query_de = ""
        tc_search_query_en = ""
        tc_args_de = ""
        tc_args_en = ""

        tracer_dict_de_keys = list(tracer_dict_de.keys())
        tracer_dict_en_keys = list(tracer_dict_en.keys())
        logger.info(f"Traced values in tracer_dict_de: {tracer_dict_de_keys}")
        logger.info(f"Traced values in tracer_dict_en: {tracer_dict_en_keys}")

        # Construct German traces
        if tracer_dict_de.get("tool_node_outputs") is not None:
            tool_node_outputs_de = tracer_dict_de.get("tool_node_outputs")
            tc_search_query_de = tool_node_outputs_de.get("search_query")
            retrieved_context_de = tool_node_outputs_de.get("tool_messages")

            last_tool_usage_de = tool_node_outputs_de.get("last_tool_usage")
            if last_tool_usage_de and "tool_calls" in last_tool_usage_de:
                raw_tools = last_tool_usage_de["tool_calls"]
                if raw_tools:
                    tool = raw_tools[0]
                    tool_name_de = tool["function"]["name"]
                    tc_args_de = json.loads(tool["function"]["arguments"])
                    tc_id_de = tool["id"]

        if tracer_dict_de.get("chatbot_response", "") is not None:
            chatbot_response_de = tracer_dict_de.get("chatbot_response", "")

        # Construct English traces
        if tracer_dict_en.get("tool_node_outputs") is not None:
            tool_node_outputs_en = tracer_dict_en.get("tool_node_outputs")
            tc_search_query_en = tool_node_outputs_en.get("search_query")
            retrieved_context_en = tool_node_outputs_en.get("tool_messages")

            last_tool_usage_en = tool_node_outputs_en.get("last_tool_usage")
            if last_tool_usage_en and "tool_calls" in last_tool_usage_en:
                raw_tools = last_tool_usage_en["tool_calls"]
                if raw_tools:
                    tool = raw_tools[0]
                    tool_name_en = tool["function"]["name"]
                    tc_args_en = json.loads(tool["function"]["arguments"])
                    tc_id_en = tool["id"]

        if tracer_dict_en.get("chatbot_response", "") is not None:
            chatbot_response_en = tracer_dict_en.get("chatbot_response", "")

        # Build row for trace results of one query_id (English and German query)
        row = {
            "query_id": faq_item.get("query_id", ""),
            "main_model": self.model_name,
            "secondary_model": self.optional_model_name,
            "user_initial_query_de": faq_item.get("user_initial_query_de", ""),
            "user_initial_query_en": faq_item.get("user_initial_query_en", ""),
            "chatbot_response_de": chatbot_response_de,
            "chatbot_response_en": chatbot_response_en,
            "graph_error_msg_de": tracer_dict_de.get("graph_error_msg", ""),
            "graph_error_msg_en": tracer_dict_en.get("graph_error_msg", ""),
            "start_time_de": tracer_dict_de.get("start_time", ""),
            "start_time_en": tracer_dict_en.get("start_time", ""),
            "end_time_de": tracer_dict_de.get("end_time", ""),
            "end_time_en": tracer_dict_en.get("end_time", ""),
            "latency_total_de": tracer_dict_de.get("latency_total", ""),
            "latency_total_en": tracer_dict_en.get("latency_total", ""),
            "last_tool_call_de": tool_name_de,
            "last_tool_call_en": tool_name_en,
            "tc_id_de": tc_id_de,
            "tc_id_en": tc_id_en,
            "tc_search_query_de": tc_search_query_de,
            "tc_search_query_en": tc_search_query_en,
            "retrieved_context_de": retrieved_context_de,
            "tracing_retrieved_context_en": retrieved_context_en,
            "tc_args_de": tc_args_de,
            "tc_args_en": tc_args_en,
            "latency_judge_node_fallback_de": tracer_dict_de.get(
                "judgement_binary_latency", ""
            ),
            "latency_judge_node_fallback_en": tracer_dict_en.get(
                "judgement_binary_latency", ""
            ),
            "latency_grade_documents_fallback_de": tracer_dict_de.get(
                "binary_score_latency", ""
            ),
            "latency_grade_documents_fallback_en": tracer_dict_en.get(
                "binary_score_latency", ""
            ),
        }
        self.traces.append(row)
        logger.info(f"[EVAL] Trace entry appended. query_id: {row['query_id']}")

    def save_to_csv(self):
        """Save data in self.traces to csv file."""
        if self.traces:
            df = pd.DataFrame(self.traces)
            file_path = f"{self.output_dir}/{self.save_path}.csv"

            df.to_csv(file_path, index=False, sep=";")
        else:
            logger.debug("[EVAL] No traces in self.traces")
        return
