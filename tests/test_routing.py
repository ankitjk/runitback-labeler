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


from labeler.routing import after_llm


def opinion(model_label, llm_label, zoom=False):
    return {"model_label": model_label, "llm_label": llm_label, "llm_zoom": zoom}


def test_agreement_is_accepted():
    assert after_llm(opinion("respawning", "respawning"), 0, 10, False) == "accept"


def test_disagreement_goes_to_the_human_when_the_context_agent_is_off():
    assert after_llm(opinion("respawning", "one_airborne"), 0, 10, False) == "human"


def test_disagreement_goes_to_the_context_agent_when_it_is_on():
    assert after_llm(opinion("respawning", "one_airborne"), 0, 10, True) == "context"


def test_an_answer_that_failed_the_checks_goes_to_the_human_not_the_context_agent():
    assert after_llm(opinion("respawning", None), 0, 10, True) == "human"


def test_a_zoom_goes_to_the_human_even_if_a_label_is_present():
    assert after_llm(opinion("respawning", "respawning", zoom=True), 0, 10, True) == "human"


def test_past_the_question_cap_the_human_route_becomes_defer():
    assert after_llm(opinion("respawning", "one_airborne"), 10, 10, False) == "defer"
    assert after_llm(opinion("respawning", None), 10, 10, False) == "defer"


def test_the_question_cap_does_not_stop_an_agreement():
    assert after_llm(opinion("respawning", "respawning"), 10, 10, False) == "accept"
