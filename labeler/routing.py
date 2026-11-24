"""Where a segment goes next. Plain functions, so the graph and the evaluation share them."""


def after_predict(segment: dict, threshold: float, never_auto_accept: frozenset[str]) -> str:
    """Return "accept" if the model's label can stand alone, otherwise "llm"."""
    if segment["model_label"] in never_auto_accept:
        return "llm"
    if segment["min_confidence"] < threshold:
        return "llm"
    return "accept"
