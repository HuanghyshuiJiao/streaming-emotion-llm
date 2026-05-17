EMOTION_UNDERSTANDING_PROMPT = """You are observing a streaming video of a person's face.
Predict only the current emotion word. Do not explain."""

EVENT_TRIGGER_PROMPT = """A meaningful emotional change may have occurred.
Generate only the emotion word for the recent stream."""

EMOTION_REASONING_PROMPT = """Predict the likely emotion and explain the visual or audio cues that support it."""

EMOTION_TOKEN_PROMPT = """Predict the person's current emotion from the recent video stream.
Answer with only one short emotion label."""
