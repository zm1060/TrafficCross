#!/usr/bin/env python3
import json
from pathlib import Path
import time
from typing import Dict, Any
import numpy as np

class MetricsLogger:
    def __init__(self, base_dir: Path, model_name: str, dataset_name: str, seed=None):
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.dataset_name = dataset_name

        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Organize metrics into model subdirectories.
        self.model_dir = self.base_dir / model_name / dataset_name
        if seed is not None:
            self.model_dir = self.model_dir / f'seed_{seed}'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Point metrics files to the model-specific directory.
        self.metrics_file = self.model_dir / f"metrics_{timestamp}.json"

        self.metrics_data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "training_metrics": [],
            "test_metrics": None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if seed is not None:
            self.metrics_data['training_seed'] = seed

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [float(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        return obj

    def log_training_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics for a training epoch."""
        epoch_data = {
            "epoch": epoch,
            "train": self._convert_to_serializable(train_metrics),
            "validation": self._convert_to_serializable(val_metrics)
        }
        self.metrics_data["training_metrics"].append(epoch_data)
        self._save_metrics()

    def log_test_metrics(self, test_metrics: Dict):
        """Log final test metrics."""
        self.metrics_data["test_metrics"] = self._convert_to_serializable(test_metrics)
        self._save_metrics()

    def _save_metrics(self):
        """Save metrics to file using a safe write pattern."""
        temp_file = self.metrics_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=4)

            # If write was successful, rename temp file to actual file
            if temp_file.exists():
                if self.metrics_file.exists():
                    self.metrics_file.unlink()
                temp_file.rename(self.metrics_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
