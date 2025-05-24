import os
import json


def test_metrics_exists():
    assert os.path.exists("metrics.json"), "Metrics metrics.json file is missing"


def test_valid_metric():
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert isinstance(metrics, dict), "dictionary missing in metrics.json"


def test_accuracy_is_valid():
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics, "accuracy field missing in metrics.json"
    accuracy = metrics["accuracy"]

    assert 0.0 <= accuracy <= 1.0, f"accuracy must be between 0.0 and 1.0, got {accuracy}"
    