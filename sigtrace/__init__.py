"""sigtrace — trace and visualise signals flowing through a torch model during inference."""

from .jepa import JEPAOutput, JEPAReport, analyze_jepa, embedding_stats
from .stats import StatsConfig, tensor_stats
from .tracer import SignalTracer, trace
from .viewer import render_html

__all__ = [
    "JEPAOutput",
    "JEPAReport",
    "SignalTracer",
    "StatsConfig",
    "analyze_jepa",
    "embedding_stats",
    "render_html",
    "tensor_stats",
    "trace",
]
__version__ = "0.1.0"
