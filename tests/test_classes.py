import itertools

import pytest

from labeler import config
from labeler.classes import CLASSES, PLAYER_STATES, scene_from_states


def test_there_are_11_classes():
    assert len(CLASSES) == 11
    assert len(set(CLASSES)) == 11


@pytest.mark.parametrize(
    "state_a, state_b, expected",
    [
        # The worked examples in CLASSES.md
        ("offstage", "on_ledge", "one_offstage"),
        ("offstage", "airborne", "one_offstage"),
        ("offstage", "offstage", "both_offstage"),
        ("on_ledge", "knocked_down", "one_on_ledge"),
        # One case per class
        ("grounded", "grounded", "both_grounded"),
        ("airborne", "grounded", "one_airborne"),
        ("airborne", "airborne", "both_airborne"),
        ("on_ledge", "on_ledge", "both_on_ledge"),
        ("knocked_down", "grounded", "one_knocked_down"),
        ("knocked_down", "airborne", "one_knocked_down"),
        ("knocked_down", "knocked_down", "both_knocked_down"),
        # Respawning is one class whether one or both players are respawning
        ("respawning", "offstage", "respawning"),
        ("respawning", "respawning", "respawning"),
    ],
)
def test_ranking(state_a, state_b, expected):
    assert scene_from_states(True, state_a, state_b) == expected
    assert scene_from_states(True, state_b, state_a) == expected


def test_not_gameplay_ignores_states():
    assert scene_from_states(False) == "non_gameplay"
    assert scene_from_states(False, "offstage", "nonsense") == "non_gameplay"


def test_every_pair_of_states_gives_a_real_class():
    for a, b in itertools.product(PLAYER_STATES, repeat=2):
        assert scene_from_states(True, a, b) in CLASSES


def test_every_class_can_be_reached():
    reached = {scene_from_states(True, a, b) for a, b in itertools.product(PLAYER_STATES, repeat=2)}
    reached.add(scene_from_states(False))
    assert reached == set(CLASSES)


@pytest.mark.parametrize("bad", ["flying", "", None, "Offstage"])
def test_unknown_state_is_refused(bad):
    with pytest.raises(ValueError):
        scene_from_states(True, bad, "grounded")


def test_never_auto_accept_names_real_classes():
    assert config.NEVER_AUTO_ACCEPT <= set(CLASSES)
