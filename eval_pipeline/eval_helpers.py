import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.core_config import settings


@dataclass
class MethodLatency:
    """Stores latency metrics for a single method."""

    latency: float = 0.0


@dataclass
class MethodOutputs:
    """Stores latency metrics for a single method."""

    outputs: Dict = field(default_factory=dict)


class EvalTracer:
    """Centralized tracer for graph methods across the agent."""

    def __init__(self):
        self._initialized = True
        self.metrics: Dict[str, MethodLatency | MethodOutputs] = {}

    def record_latency(self, method_name: str, start: float, end: float):
        """Record latency for a graph method."""
        latency = end - start
        self.metrics[method_name] = MethodLatency(
            latency=latency,
        )

    def record_output(self, method_name: str, outputs: dict):
        """Record output of a graph method."""
        self.metrics[method_name] = MethodOutputs(
            outputs=outputs,
        )

    def get_attr_dict(self) -> Dict[str, Any]:
        """Return metrics as dict. Separate keys for latency and outputs."""
        result = {}
        for method_name, metric in self.metrics.items():
            if type(metric) == MethodLatency:
                result[f"{method_name}_latency"] = metric.latency
            elif type(metric) == MethodOutputs:
                result[f"{method_name}_outputs"] = metric.outputs

        return result

    def reset(self):
        """Clear all metrics."""
        self.metrics.clear()


def trace_method(
    method_name: Optional[str] = None, trace_type: str = "latency"
) -> Callable:
    """
    Decorator: traces latency or outputs.
    example: @trace_method("tool_node", trace_type="outputs")

    Args:
        method_name: Name for storing. Defaults to func.__name__.
        trace_type: "latency" or "outputs".
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Check if tracing is enabled in settings
            if not settings.eval_mode_enabled:
                return func(self, *args, **kwargs)

            # Get or create tracer on instance
            if not hasattr(self, "_eval_tracer"):
                self._eval_tracer = EvalTracer()

            tracer = self._eval_tracer
            name = method_name or func.__name__

            # Measure latency for method
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            end = time.perf_counter()

            # Record conditionally
            if trace_type == "latency":
                tracer.record_latency(name, start, end)
            if trace_type == "outputs":
                if type(result) == dict:
                    tracer.record_output(name, result)
                else:
                    tracer.record_output(
                        name, {"result": result}
                    )  # make sure method output is dict

            return result

        return wrapper

    return decorator


# Load traces from csv file
def load_traces(trace_path: str) -> List[Dict[str, Any]]:
    """Convert a CSV of conversation traces to list of dicts."""
    df = pd.read_csv(trace_path, sep=";")  # Load cvs into DataFrame
    return df.to_dict(orient="records")
