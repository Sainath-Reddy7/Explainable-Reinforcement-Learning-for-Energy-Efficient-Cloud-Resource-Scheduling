"""
build_dashboard.py — Inject results JSON + base64 images into the HTML template.

Produces results/dashboard.html — a self-contained operator console that can be
opened in any browser without a server.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "results"
TEMPLATE = HERE / "dashboard_template.html"
OUT = R / "dashboard.html"


def _img_b64(path):
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def main():
    payload = {
        "comparison": json.load(open(R / "comparison_results.json")),
        "trust": json.load(open(R / "trust_metrics.json")),
        "decision_log": json.load(open(R / "decision_log.json")),
        "history": json.load(open(R / "dqn_training_history.json")),
        "manifest": json.load(open(R / "manifest.json")) if (R / "manifest.json").exists() else None,
        "images": {
            "scheduler": _img_b64(R / "scheduler_comparison.png"),
            "learning": _img_b64(R / "dqn_learning_curve.png"),
            "xai": _img_b64(R / "xai_method_comparison.png"),
            "latency": _img_b64(R / "xai_latency.png"),
        },
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("{{DATA}}", json.dumps(payload))
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
