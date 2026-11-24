import pytest

from labeler.inference import class_names_from_mapping


def test_order_comes_from_the_mapping():
    mapping = {"1": "both_grounded", "0": "respawning", "2": "non_gameplay"}
    assert class_names_from_mapping(mapping, 3) == ["respawning", "both_grounded", "non_gameplay"]


def test_wrong_number_of_classes_is_refused():
    with pytest.raises(ValueError, match="2 classes but the model has 3 outputs"):
        class_names_from_mapping({"0": "respawning", "1": "both_grounded"}, 3)


def test_unknown_class_name_is_refused():
    with pytest.raises(ValueError, match="does not know"):
        class_names_from_mapping({"0": "respawning", "1": "flying"}, 2)
