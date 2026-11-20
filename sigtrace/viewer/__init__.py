"""Static HTML viewer for sigtrace traces."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

PLACEHOLDER = "/*__SIGTRACE_DATA__*/null"


def template() -> str:
    return resources.files(__package__).joinpath("index.html").read_text(encoding="utf-8")


def render_html(trace: dict[str, Any] | str | None) -> str:
    """Return the viewer HTML with `trace` embedded (or empty, for drag-and-drop use)."""
    html = template()
    if trace is None:
        return html
    data = trace if isinstance(trace, str) else json.dumps(trace, separators=(",", ":"), allow_nan=False)
    # Neutralise sequences that could terminate the <script> element early.
    data = data.replace("</", "<\\/").replace("<!--", "<\\!--")
    return html.replace(PLACEHOLDER, data, 1)
