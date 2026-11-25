"""The checks on the LLM's answer, and the prompt. No network: the Gemini call itself is not tested here."""

import json

import pytest

from labeler.llm import MAX_REASON_CHARS, _section, build_prompt, check_answer


def answer(**fields):
    base = {"gameplay": True, "finishing_blow_zoom": False, "state_a": "grounded", "state_b": "grounded", "reason": "r"}
    return json.dumps({**base, **fields})


def test_code_works_out_the_class_from_the_two_states():
    result = check_answer(answer(state_a="offstage", state_b="on_ledge", reason="one below the edge"))
    assert result == {"llm_label": "one_offstage", "llm_zoom": False, "llm_reason": "one below the edge", "llm_error": None}


def test_not_gameplay_needs_no_states():
    text = json.dumps({"gameplay": False, "finishing_blow_zoom": False, "reason": "results screen"})
    assert check_answer(text)["llm_label"] == "non_gameplay"


def test_a_zoom_has_no_class():
    result = check_answer(answer(finishing_blow_zoom=True))
    assert result["llm_label"] is None and result["llm_zoom"] is True and result["llm_error"] is None


@pytest.mark.parametrize("text", [
    "not json",
    "[1, 2]",
    answer(state_a="flying"),                       # a state that does not exist
    answer(state_b=None),                           # gameplay with a state missing
    answer(gameplay="yes"),                         # not true/false
    answer(reason=5),
    json.dumps({"gameplay": True, "reason": "r"}),  # zoom flag missing
    None,
])
def test_a_bad_answer_has_no_class_and_says_why(text):
    result = check_answer(text)
    assert result["llm_label"] is None
    assert result["llm_zoom"] is False
    assert result["llm_error"]


def test_the_llm_cannot_name_a_class_directly():
    # A "class" field is ignored: only the states count.
    result = check_answer(answer(state_a="airborne", state_b="grounded", **{"class": "respawning"}))
    assert result["llm_label"] == "one_airborne"


def test_a_long_reason_is_cut():
    assert len(check_answer(answer(reason="x" * 5000))["llm_reason"]) == MAX_REASON_CHARS


def test_the_prompt_takes_its_definitions_from_classes_md():
    classes_md = "# T\n\n## Terms\n\nTERMS TEXT\n\n## Classes\n\nCLASS TEXT\n\n## For this labeller\n\nPRIVATE\n"
    prompt = build_prompt(classes_md)
    assert "TERMS TEXT" in prompt and "CLASS TEXT" in prompt
    assert "PRIVATE" not in prompt
    assert "knocked_down" in prompt  # the state names come from classes.py


def test_the_real_classes_md_has_the_sections_the_prompt_needs():
    assert "Offstage" in build_prompt()


def test_a_missing_section_is_an_error():
    with pytest.raises(ValueError):
        _section("## Terms\n\nx\n", "Classes")


def test_a_model_id_that_could_leave_the_cache_folder_is_refused(monkeypatch, tmp_path):
    from labeler.llm import GeminiLabeler

    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    with pytest.raises(ValueError):
        GeminiLabeler(model="../../somewhere", cache_dir=tmp_path)


def test_no_key_in_the_environment_is_an_error(monkeypatch, tmp_path):
    from labeler.llm import GeminiLabeler

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiLabeler(cache_dir=tmp_path)
