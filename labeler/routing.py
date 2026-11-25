"""Where a segment goes next. Plain functions, so the graph and the evaluation share them."""


def after_predict(segment: dict, threshold: float, never_auto_accept: frozenset[str]) -> str:
    """Return "accept" if the model's label can stand alone, otherwise "llm"."""
    if segment["model_label"] in never_auto_accept:
        return "llm"
    if segment["min_confidence"] < threshold:
        return "llm"
    return "accept"


def after_llm(segment: dict, questions_asked: int, max_questions: int, use_context_agent: bool) -> str:
    """Where a segment goes once the first LLM has answered: "accept", "context", "human" or "defer"."""
    to_human = "human" if questions_asked < max_questions else "defer"
    if segment.get("llm_zoom") or segment.get("llm_label") is None:
        return to_human  # a zoomed finishing blow, or an answer that failed the checks
    if segment["llm_label"] == segment["model_label"]:
        return "accept"
    return "context" if use_context_agent else to_human
