"""Analytical governance rules for the speed-dating case study."""

from __future__ import annotations

from dataclasses import dataclass

POST_OUTCOME_FIELDS = {
    "match",
    "dec_o",
    "like_o",
    "prob_o",
    "met_o",
}


@dataclass(frozen=True)
class FeatureAudit:
    requested: tuple[str, ...]
    forbidden: tuple[str, ...]
    accepted: tuple[str, ...]


def audit_prediction_features(
    features: list[str] | tuple[str, ...] | set[str],
    target: str = "match",
) -> FeatureAudit:
    """Fail closed when outcome or post-outcome information enters a realistic model."""

    requested = tuple(dict.fromkeys(str(feature) for feature in features))
    forbidden_set = POST_OUTCOME_FIELDS | {target}
    forbidden = tuple(sorted(feature for feature in requested if feature in forbidden_set))
    accepted = tuple(feature for feature in requested if feature not in forbidden_set)

    if forbidden:
        raise ValueError(
            "Prediction feature leakage detected: "
            + ", ".join(forbidden)
        )

    return FeatureAudit(
        requested=requested,
        forbidden=(),
        accepted=accepted,
    )


def assert_binary_target(
    values: list[object] | tuple[object, ...] | set[object],
) -> None:
    """Require a strict binary target before statistical or predictive modelling."""

    unique = {int(value) for value in values}
    if not unique.issubset({0, 1}):
        raise ValueError(f"Target contains non-binary values: {sorted(unique)}")


def assert_probability(value: float) -> None:
    """Validate model/dashboard probabilities before presentation."""

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"Probability outside [0, 1]: {value}")
