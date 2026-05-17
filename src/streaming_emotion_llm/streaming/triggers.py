"""Event-trigger policies for online response generation."""


class TemporalEventTrigger:
    def __init__(self, min_confidence: float = 0.5, cooldown_seconds: float = 2.0):
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger_time = None

    def should_trigger(self, timestamp_sec: float, confidence: float) -> bool:
        if confidence < self.min_confidence:
            return False
        if self._last_trigger_time is None:
            self._last_trigger_time = timestamp_sec
            return True
        if timestamp_sec - self._last_trigger_time >= self.cooldown_seconds:
            self._last_trigger_time = timestamp_sec
            return True
        return False
