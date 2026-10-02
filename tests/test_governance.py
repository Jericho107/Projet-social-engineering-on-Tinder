import pytest

from src.governance import (
    assert_binary_target,
    assert_probability,
    audit_prediction_features,
)


def test_realistic_features_are_accepted() -> None:
    result = audit_prediction_features(
        ["age", "age_o", "attr", "fun", "intel", "age_gap", "int_corr"]
    )
    assert result.forbidden == ()
    assert "age_gap" in result.accepted


def test_target_leakage_fails_closed() -> None:
    with pytest.raises(ValueError, match="leakage"):
        audit_prediction_features(["age", "match", "attr"])


def test_partner_post_outcome_signal_fails_closed() -> None:
    with pytest.raises(ValueError, match="dec_o"):
        audit_prediction_features(["age", "dec_o"])


def test_binary_target_contract() -> None:
    assert_binary_target([0, 1, 1, 0])
    with pytest.raises(ValueError):
        assert_binary_target([0, 1, 2])


def test_probability_contract() -> None:
    assert_probability(0.51)
    with pytest.raises(ValueError):
        assert_probability(1.01)
