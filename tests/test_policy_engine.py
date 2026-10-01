from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from policy_engine.exceptions import PolicyConfigurationError
from policy_engine.loader import load_policy_set
from policy_engine.matcher import evaluate_policies
from policy_engine.models import (
    EvaluationStatus,
    Policy,
    PolicyRequest,
)
from policy_engine.taxonomy import Activity, Audience, normalize_activity
from policy_engine.weather import NormalizedWeather, WeatherCondition, normalize_wmo_code


ROOT = __file__
MANIFEST_PATH = __import__("pathlib").Path(ROOT).parents[1] / "policies" / "manifest.yaml"
AS_OF = date(2026, 10, 2)


@pytest.fixture(scope="module")
def policy_set():
    return load_policy_set(MANIFEST_PATH)


def policy_record(
    policy_id="TEST-WIND",
    *,
    severity="moderate",
    priority=10,
    conditions=None,
    activity="cycling",
    conflicts_with=None,
    version="1.0.0",
):
    return {
        "policy_id": policy_id,
        "version": version,
        "category": "outdoor_exercise",
        "activities": [activity],
        "severity": severity,
        "priority": priority,
        "effective_from": "2026-01-01",
        "effective_until": None,
        "conditions": conditions or {
            "all": [
                {
                    "field": "weather.wind_speed_10m",
                    "operator": "gte",
                    "value": 30,
                    "unit": "km/h",
                }
            ]
        },
        "directive": f"Approved directive for {policy_id}.",
        "rationale": "A test rationale grounded in the supplied condition.",
        "source": "Test policy set",
        "conflicts_with": conflicts_with or [],
    }


def load_records(tmp_path, records, *, policy_version="1.0.0", manifest_version=None):
    manifest_path = tmp_path / "manifest.yaml"
    document_path = tmp_path / "policies.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "policy_set_version": manifest_version or policy_version,
                "updated_at": "2026-10-02",
                "description": "Test policy set",
                "policies_file": "policies.yaml",
            }
        ),
        encoding="utf-8",
    )
    document_path.write_text(
        yaml.safe_dump({"policy_set_version": policy_version, "sops": records}),
        encoding="utf-8",
    )
    return load_policy_set(manifest_path)


def evaluate(policy_set, *, activity=Activity.CYCLING, weather=None, audience=None):
    return evaluate_policies(
        PolicyRequest(activity=activity, audience=audience),
        weather or NormalizedWeather(wind_speed_10m=45),
        policy_set,
        as_of=AS_OF,
    )


def test_exact_numeric_match_records_condition_evidence(policy_set):
    result = evaluate(policy_set, weather=NormalizedWeather(wind_speed_10m=45))

    match = next(match for match in result.matches if match.policy_id == "OUTDOOR-EXERCISE-WIND-STRONG")
    evidence = match.condition_results[0]
    assert result.status == EvaluationStatus.MATCH
    assert evidence.field == "weather.wind_speed_10m"
    assert evidence.operator.value == "gte"
    assert evidence.threshold == 40
    assert evidence.actual == 45
    assert evidence.unit == "km/h"
    assert evidence.matched is True
    assert match.directive == "Postpone exposed cycling or exercise routes until winds ease."


def test_numeric_non_match(policy_set):
    result = evaluate(policy_set, weather=NormalizedWeather(wind_speed_10m=24.9))

    assert result.status == EvaluationStatus.NO_MATCH
    assert result.matches == ()


def test_all_conditions_require_every_condition(policy_set, tmp_path):
    record = policy_record(
        conditions={
            "all": [
                {"field": "weather.wind_speed_10m", "operator": "gte", "value": 30, "unit": "km/h"},
                {"field": "weather.temperature_2m", "operator": "gt", "value": 20, "unit": "degC"},
            ]
        }
    )
    custom_set = load_records(tmp_path, records=[record])
    result = evaluate(
        custom_set,
        weather=NormalizedWeather(wind_speed_10m=35, temperature_2m=21),
    )

    assert result.status == EvaluationStatus.MATCH
    assert len(result.matches[0].condition_results) == 2
    assert all(item.matched for item in result.matches[0].condition_results)


def test_any_conditions_match_one_or_more(policy_set, tmp_path):
    record = policy_record(
        conditions={
            "any": [
                {"field": "weather.wind_speed_10m", "operator": "gte", "value": 40, "unit": "km/h"},
                {"field": "weather.uv_index", "operator": "gte", "value": 8, "unit": "index"},
            ]
        }
    )
    custom_set = load_records(tmp_path, records=[record])
    result = evaluate(custom_set, weather=NormalizedWeather(wind_speed_10m=10, uv_index=9))

    assert result.status == EvaluationStatus.MATCH
    assert [item.matched for item in result.matches[0].condition_results] == [False, True]


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("eq", 10, True),
        ("neq", 11, True),
        ("gt", 9, True),
        ("gte", 10, True),
        ("lt", 11, True),
        ("lte", 10, True),
        ("in", [9, 10], True),
        ("not_in", [11, 12], True),
    ],
)
def test_supported_numeric_operators(operator, value, expected, tmp_path):
    record = policy_record(
        conditions={
            "all": [
                {
                    "field": "weather.wind_speed_10m",
                    "operator": operator,
                    "value": value,
                    "unit": "km/h",
                }
            ]
        }
    )
    result = evaluate(
        load_records(tmp_path, [record]),
        weather=NormalizedWeather(wind_speed_10m=10),
    )

    assert (result.status == EvaluationStatus.MATCH) is expected


def test_activity_mismatch_does_not_match(policy_set):
    result = evaluate(policy_set, activity=Activity.PICNIC, weather=NormalizedWeather(wind_speed_10m=45))

    assert result.status == EvaluationStatus.NO_MATCH


def test_multiple_matching_sops_are_all_returned(policy_set):
    result = evaluate(policy_set, weather=NormalizedWeather(wind_speed_10m=45))
    matching_ids = {match.policy_id for match in result.matches}

    assert "OUTDOOR-EXERCISE-WIND-BREEZY" in matching_ids
    assert "OUTDOOR-EXERCISE-WIND-STRONG" in matching_ids
    assert len(result.matches) == 2


def test_severity_ordering_precedes_priority(tmp_path):
    records = [
        policy_record("TEST-LOW", severity="low", priority=100),
        policy_record("TEST-HIGH", severity="high", priority=1),
    ]
    result = evaluate(load_records(tmp_path, records))

    assert [match.policy_id for match in result.matches] == ["TEST-HIGH", "TEST-LOW"]


def test_priority_ordering_breaks_equal_severity_ties(tmp_path):
    records = [
        policy_record("TEST-LOW-PRIORITY", severity="high", priority=1),
        policy_record("TEST-HIGH-PRIORITY", severity="high", priority=5),
    ]
    result = evaluate(load_records(tmp_path, records))

    assert [match.policy_id for match in result.matches] == ["TEST-HIGH-PRIORITY", "TEST-LOW-PRIORITY"]


def test_policy_id_breaks_remaining_order_tie(tmp_path):
    records = [
        policy_record("TEST-Z", severity="high", priority=5),
        policy_record("TEST-A", severity="high", priority=5),
    ]
    result = evaluate(load_records(tmp_path, records))

    assert [match.policy_id for match in result.matches] == ["TEST-A", "TEST-Z"]


def test_no_match_is_structured_and_contains_no_advice(policy_set):
    result = evaluate(policy_set, weather=NormalizedWeather(wind_speed_10m=1))

    assert result.status == EvaluationStatus.NO_MATCH
    assert result.matches == ()
    assert result.conflicts == ()


@pytest.mark.parametrize(
    "condition, message",
    [
        (
            {"field": "weather.untrusted.path", "operator": "gte", "value": 10, "unit": "km/h"},
            "Unknown weather field",
        ),
        (
            {"field": "weather.wind_speed_10m", "operator": "approx", "value": 10, "unit": "km/h"},
            "operator",
        ),
    ],
)
def test_invalid_field_and_operator_are_rejected(condition, message):
    with pytest.raises(ValidationError, match=message):
        Policy.model_validate(policy_record(conditions={"all": [condition]}))


def test_invalid_severity_is_rejected():
    with pytest.raises(ValidationError, match="severity"):
        Policy.model_validate(policy_record(severity="extreme"))


def test_duplicate_policy_ids_are_rejected(tmp_path):
    record = policy_record("TEST-DUPLICATE")

    with pytest.raises(PolicyConfigurationError, match="Duplicate policy_id"):
        load_records(tmp_path, [record, record.copy()])


def test_invalid_activity_is_rejected():
    with pytest.raises(ValidationError, match="activities"):
        Policy.model_validate(policy_record(activity="unlisted_activity"))


def test_malformed_yaml_fails_clearly(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("policy_set_version: [unterminated", encoding="utf-8")

    with pytest.raises(PolicyConfigurationError, match="Malformed YAML"):
        load_policy_set(manifest_path)


def test_duplicate_yaml_mapping_key_fails_clearly(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "policy_set_version: '1.0.0'\n"
        "policy_set_version: '2.0.0'\n"
        "updated_at: '2026-10-02'\n"
        "description: test\n"
        "policies_file: 'policies.yaml'\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError, match="Duplicate YAML mapping key"):
        load_policy_set(manifest_path)


@pytest.mark.parametrize("version", ["1", "v1.0", "1.0.x", ""])
def test_invalid_policy_version_is_rejected(version):
    with pytest.raises(ValidationError, match="version"):
        Policy.model_validate(policy_record(version=version))


def test_policy_set_version_mismatch_is_rejected(tmp_path):
    with pytest.raises(PolicyConfigurationError, match="version mismatch"):
        load_records(tmp_path, [policy_record()], policy_version="1.0.0", manifest_version="2.0.0")


def test_unresolved_policy_conflict_returns_conflict_and_both_matches(tmp_path):
    first = policy_record("TEST-FIRST", conflicts_with=["TEST-SECOND"])
    second = policy_record("TEST-SECOND", conflicts_with=["TEST-FIRST"])
    result = evaluate(load_records(tmp_path, [first, second]))

    assert result.status == EvaluationStatus.POLICY_CONFLICT
    assert {match.policy_id for match in result.matches} == {"TEST-FIRST", "TEST-SECOND"}
    assert result.conflicts[0].policy_ids == ("TEST-FIRST", "TEST-SECOND")


def test_conflict_declarations_must_be_reciprocal(tmp_path):
    first = policy_record("TEST-FIRST", conflicts_with=["TEST-SECOND"])
    second = policy_record("TEST-SECOND")

    with pytest.raises(PolicyConfigurationError, match="reciprocal"):
        load_records(tmp_path, [first, second])


def test_categorical_fuzzy_policy_uses_deterministic_wmo_mapping(policy_set):
    assert normalize_wmo_code(95) == WeatherCondition.THUNDERSTORM
    result = evaluate(
        policy_set,
        activity=Activity.TRAVEL,
        weather=NormalizedWeather(wmo_condition=normalize_wmo_code(95)),
    )

    assert result.status == EvaluationStatus.MATCH
    assert result.matches[0].policy_id == "TRAVEL-THUNDERSTORM"
    assert result.matches[0].condition_results[0].actual == "thunderstorm"


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("bike ride", Activity.CYCLING),
        ("biking", Activity.CYCLING),
        ("jog", Activity.RUNNING),
        ("outdoor_play", Activity.OUTDOOR_PLAY),
    ],
)
def test_activity_alias_normalization(phrase, expected):
    assert normalize_activity(phrase) == expected


def test_unknown_activity_alias_is_rejected():
    with pytest.raises(ValueError, match="Unsupported activity"):
        normalize_activity("extreme adventure")


def test_new_policy_requires_no_code_change(tmp_path):
    new_policy = policy_record(
        "OUTDOOR-EXERCISE-WIND-VERY-STRONG",
        severity="critical",
        priority=90,
        conditions={
            "all": [
                {
                    "field": "weather.wind_speed_10m",
                    "operator": "gte",
                    "value": 55,
                    "unit": "km/h",
                }
            ]
        },
    )
    policy_set = load_records(tmp_path, [new_policy])
    result = evaluate(policy_set, weather=NormalizedWeather(wind_speed_10m=56))

    assert [match.policy_id for match in result.matches] == ["OUTDOOR-EXERCISE-WIND-VERY-STRONG"]
    assert result.matches[0].condition_results[0].actual == 56


def test_unknown_weather_unit_is_rejected():
    record = policy_record(
        conditions={
            "all": [
                {"field": "weather.wind_speed_10m", "operator": "gte", "value": 30, "unit": "m/s"}
            ]
        }
    )

    with pytest.raises(ValidationError, match="canonical unit"):
        Policy.model_validate(record)


def test_missing_weather_value_is_evidenced_and_never_matches(tmp_path):
    custom_set = load_records(tmp_path, [policy_record()])
    result = evaluate(custom_set, weather=NormalizedWeather())

    assert result.status == EvaluationStatus.NO_MATCH


def test_policy_audience_is_required_when_policy_is_audience_scoped(policy_set):
    no_audience = evaluate(
        policy_set,
        activity=Activity.OUTDOOR_PLAY,
        weather=NormalizedWeather(uv_index=7),
    )
    with_audience = evaluate(
        policy_set,
        activity=Activity.OUTDOOR_PLAY,
        audience=Audience.CHILDREN,
        weather=NormalizedWeather(uv_index=7),
    )

    assert no_audience.status == EvaluationStatus.NO_MATCH
    assert any(match.policy_id == "CHILDREN-OUTDOOR-PLAY-UV" for match in with_audience.matches)


def test_effective_dates_are_enforced(tmp_path):
    record = policy_record()
    record["effective_until"] = "2026-10-01"
    custom_set = load_records(tmp_path, [record])

    assert evaluate(custom_set).status == EvaluationStatus.NO_MATCH


def test_policy_manifest_and_document_load(policy_set):
    assert policy_set.manifest.policy_set_version == "1.0.0"
    assert len(policy_set.policies) == 12


def test_manifest_rejects_path_traversal(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "policy_set_version: '1.0.0'\n"
        "updated_at: '2026-10-02'\n"
        "description: test\n"
        "policies_file: '../outside.yaml'\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigurationError):
        load_policy_set(manifest_path)