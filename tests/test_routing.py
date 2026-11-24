from labeler.routing import after_predict

NEVER = frozenset({"respawning", "one_on_ledge"})


def seg(label, confidence):
    return {"model_label": label, "min_confidence": confidence}


def test_confident_segment_is_accepted():
    assert after_predict(seg("both_grounded", 0.9), 0.6, NEVER) == "accept"


def test_confidence_equal_to_threshold_is_accepted():
    assert after_predict(seg("both_grounded", 0.6), 0.6, NEVER) == "accept"


def test_unsure_segment_goes_to_the_llm():
    assert after_predict(seg("both_grounded", 0.59), 0.6, NEVER) == "llm"


def test_over_predicted_classes_always_go_to_the_llm():
    assert after_predict(seg("respawning", 0.99), 0.6, NEVER) == "llm"
    assert after_predict(seg("one_on_ledge", 0.99), 0.6, NEVER) == "llm"
