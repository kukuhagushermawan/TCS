"""Discovers available TCS YOLO models (bundled + fine-tuned) and their saved
accuracy metrics, so the UI can list them without re-running training/eval.

Convention: every model file in models/*.pt may have a sibling metrics report
written by training/train_tcs.py at the same stem + ".metrics.json" - e.g.
models/tcs_palm_yolov8n_v2_20260708.pt -> models/tcs_palm_yolov8n_v2_20260708.metrics.json
"""
from __future__ import annotations

import json
import re
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

    def summary(self) -> str:
        """One-line accuracy summary for dropdown/status display."""
        if not self.metrics:
            return "Belum ada laporan metrik."
        m = self.metrics.get("metrics", {}) or {}
        parts = []
        if "precision" in m:
            parts.append(f"P={m['precision']:.3f}")
        if "recall" in m:
            parts.append(f"R={m['recall']:.3f}")
        if "map50" in m:
            parts.append(f"mAP50={m['map50']:.3f}")
        if "map50_95" in m:
            parts.append(f"mAP50-95={m['map50_95']:.3f}")
        text = ", ".join(parts) if parts else "metrik tidak lengkap"
        trained_at = self.metrics.get("trained_at")
        return text + (f" (dilatih {trained_at})" if trained_at else "")

    def detail_text(self) -> str:
        """Full multi-line summary for a details view."""
        if not self.metrics:
            return f"{self.name}\n\nBelum ada laporan metrik tersimpan untuk model ini."
        m = self.metrics
        acc = m.get("metrics", {}) or {}
        det = m.get("val_detection_summary", {}) or {}
        lines = [
            f"Model: {self.name}",
            f"Base model: {m.get('base_model', '-')}",
            f"Dataset: {m.get('dataset', '-')}",
            f"Dilatih pada: {m.get('trained_at', '-')}",
            f"Epoch dijalankan: {m.get('epochs_run', '-')}",
            "",
            f"Precision: {acc.get('precision', float('nan')):.4f}" if "precision" in acc else "Precision: -",
            f"Recall: {acc.get('recall', float('nan')):.4f}" if "recall" in acc else "Recall: -",
            f"mAP@0.5: {acc.get('map50', float('nan')):.4f}" if "map50" in acc else "mAP@0.5: -",
            f"mAP@0.5:0.95: {acc.get('map50_95', float('nan')):.4f}" if "map50_95" in acc else "mAP@0.5:0.95: -",
        ]
        if det:
            lines += [
                "",
                "Ringkasan deteksi pada validation set:",
                f"  Ground truth : {det.get('total_ground_truth', '-')}",
                f"  Terdeteksi   : {det.get('total_detected', '-')}",
                f"  True Positive: {det.get('true_positive', '-')}",
                f"  False Positive: {det.get('false_positive', '-')}",
                f"  False Negative: {det.get('false_negative', '-')}",
            ]
        return "\n".join(lines)


def discover_models() -> List[ModelInfo]:
    """List every models/*.pt, newest-trained first, each with its metrics if present."""
    directory = models_dir()
    if not directory.exists():
        return []
    infos: List[ModelInfo] = []
    for pt_path in sorted(directory.glob("*.pt")):
        metrics = None
        m_path = metrics_path_for(pt_path)
        if m_path.exists():
            try:
                metrics = json.loads(m_path.read_text(encoding="utf-8"))
            except Exception:
                metrics = None
        infos.append(ModelInfo(path=pt_path, metrics=metrics))
    # Most accurate model first (by mAP@0.5:0.95, the strictest/most reliable
    # figure), not just the most recently trained - a newer run isn't
    # necessarily better (e.g. fine-tuning on the same dataset again, or a
    # bigger architecture that didn't help). Models without a metrics report
    # sort last since they can't be compared.
    def _score(info: ModelInfo) -> float:
        metrics = (info.metrics or {}).get("metrics") or {}
        return metrics.get("map50_95", -1.0)

    infos.sort(key=_score, reverse=True)
    return infos


def infer_arch_tag(base_model_name: str) -> str:
    """Guess the YOLOv8 size variant (n/s/m/l/x) from a base model's filename.

    Used so a fine-tuned file's name reflects what it actually is (e.g.
    "tcs_palm_yolov8s_v1...") instead of always saying "yolov8n" regardless
    of which architecture was actually trained.
    """
    match = re.search(r"(yolov8[nsmlx])", base_model_name.lower())
    return match.group(1) if match else "yolov8n"


def next_version_number(arch_tag: str = "yolov8n", directory: Optional[Path] = None) -> int:
    """Return the next 'vN' index for a new fine-tuned model file name.

    Versioned per architecture tag, so "tcs_palm_yolov8n_v*" and
    "tcs_palm_yolov8s_v*" each have their own independent counter.
    """
    directory = directory or models_dir()
    if not directory.exists():
        return 2
    max_v = 1
    for pt_path in directory.glob(f"tcs_palm_{arch_tag}_v*.pt"):
        match = re.search(r"_v(\d+)_", pt_path.name)
        if match:
            max_v = max(max_v, int(match.group(1)))
    return max_v + 1
