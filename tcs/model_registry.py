"""Discovers TCS's bundled detection model(s) and their saved accuracy
metrics, so the pipeline can pick one without re-running training/eval.

Convention: every model file in models/*.onnx may have a sibling metrics
report at the same stem + ".metrics.json" - e.g.
models/tcs_palm_detector.onnx -> models/tcs_palm_detector.metrics.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def models_dir() -> Path:
    from app.resources import resource_path

    return resource_path("models")


def metrics_path_for(model_path: Path) -> Path:
    return model_path.parent / (model_path.stem + ".metrics.json")


@dataclass
class ModelInfo:
    path: Path
    metrics: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return self.path.name


def discover_models() -> List[ModelInfo]:
    """List every models/*.onnx, most accurate first, each with its metrics if present."""
    directory = models_dir()
    if not directory.exists():
        return []
    infos: List[ModelInfo] = []
    for model_path in sorted(directory.glob("*.onnx")):
        metrics = None
        m_path = metrics_path_for(model_path)
        if m_path.exists():
            try:
                metrics = json.loads(m_path.read_text(encoding="utf-8"))
            except Exception:
                metrics = None
        infos.append(ModelInfo(path=model_path, metrics=metrics))
    # Most accurate model first (by mAP@0.5:0.95, the strictest/most reliable
    # figure) when more than one is bundled. Models without a metrics report
    # sort last since they can't be compared.
    def _score(info: ModelInfo) -> float:
        metrics = (info.metrics or {}).get("metrics") or {}
        return metrics.get("map50_95", -1.0)

    infos.sort(key=_score, reverse=True)
    return infos
