"""Metrics for emotion prediction, transition detection, triggers, and latency."""


def compute_metrics(predictions: list[dict], references: list[dict]) -> dict:
    raise NotImplementedError("Implement evaluation metrics here.")
