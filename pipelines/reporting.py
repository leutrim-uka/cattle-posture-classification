from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np


def summarize_run_metrics(run_metrics: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not run_metrics:
        return {}

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    model_names = run_metrics[0].keys()

    for model_name in model_names:
        metric_keys = run_metrics[0][model_name].keys()
        summary[model_name] = {}
        for metric_key in metric_keys:
            vals = np.array([rm[model_name][metric_key] for rm in run_metrics], dtype=float)
            summary[model_name][metric_key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    return summary


def print_summary(title: str, summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for model_name, metric_map in summary.items():
        print(f"\nModel: {model_name}")
        for metric_name, stats in metric_map.items():
            print(f"  {metric_name:25s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print("=" * 60)


def save_summary(
    out_path: str,
    title: str,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    run_metrics: List[Dict[str, Dict[str, float]]] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload: Dict[str, Any] = {
        "title": title,
        "summary": summary,
    }
    if run_metrics is not None:
        payload["runs"] = run_metrics
    if extra is not None:
        payload["extra"] = extra

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
