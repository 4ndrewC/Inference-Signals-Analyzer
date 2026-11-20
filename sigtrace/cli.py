"""Command line interface: ``python -m sigtrace {html,serve,demo}``."""

from __future__ import annotations

import argparse
import gzip
import http.server
import threading
import webbrowser
from pathlib import Path

from .viewer import render_html


def _read_trace(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def cmd_html(args: argparse.Namespace) -> int:
    src = Path(args.trace)
    out = Path(args.output) if args.output else src.with_name(src.name.split(".")[0] + ".html")
    out.write_text(render_html(_read_trace(src)), encoding="utf-8")
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    page = render_html(_read_trace(Path(args.trace)) if args.trace else None).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *a) -> None:  # quiet
            pass

    srv = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print(f"serving sigtrace viewer at {url}  (Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    out = Path(args.output or f"sigtrace-demo-{args.model}.html")
    if args.model.startswith("jepa"):
        from .demo_jepa import run_jepa_demo

        print("training a small JEPA world model (a few seconds)...")
        report = run_jepa_demo(collapse=args.model == "jepa-collapsed", horizon=args.horizon)
        report.save_html(out)
        print(report.summary())
        print(f"-> {out}")
    else:
        tracer = run_demo(args.model, steps=args.steps)
        tracer.save_html(out)
        print(f"traced {len(tracer.events)} module calls over {len(tracer.steps)} step(s) -> {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sigtrace", description="Inference signal tracer for torch modules.")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("html", help="build a standalone HTML viewer from a trace .json(.gz)")
    h.add_argument("trace")
    h.add_argument("-o", "--output")
    h.add_argument("--open", action="store_true", help="open in the browser")
    h.set_defaults(fn=cmd_html)

    s = sub.add_parser("serve", help="serve the viewer over HTTP (for remote boxes)")
    s.add_argument("trace", nargs="?", help="trace to embed (omit for drag-and-drop)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true")
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("demo", help="trace a built-in demo model")
    d.add_argument("--model", choices=["mlp", "cnn", "transformer", "jepa", "jepa-collapsed"], default="transformer")
    d.add_argument("--steps", type=int, default=8, help="generation steps (transformer)")
    d.add_argument("--horizon", type=int, default=4, help="rollout horizon (jepa)")
    d.add_argument("-o", "--output")
    d.add_argument("--open", action="store_true")
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)
