from datetime import date
from enum import Enum
from typing import Any

from policy_engine.fields import WEATHER_FIELD_SPECS
from policy_engine.models import (
    Condition,
    ConditionEvidence,
    Conditions,
    EvaluationStatus,
    Operator,
    Policy,
    PolicyConflict,
    PolicyEvaluationResult,
    PolicyMatch,
    PolicyRequest,
    PolicySet,
    SEVERITY_RANK,
)
from policy_engine.weather import NormalizedWeather


def _condition_evidence(
    group: str,
    condition: Condition,
    weather: NormalizedWeather,
) -> ConditionEvidence:
    field_spec = WEATHER_FIELD_SPECS[condition.field]
    raw_actual: Any = getattr(weather, field_spec.attribute)
    actual = raw_actual.value if isinstance(raw_actual, Enum) else raw_actual
    missing = actual is None
    matched = False if missing else _compare(actual, condition.operator, condition.value)
    expression = f"{condition.field} {condition.operator.value} {condition.value}"
    return ConditionEvidence(
        group=group,
        field=condition.field,
        operator=condition.operator,
        threshold=condition.value,
        actual=actual,
        unit=field_spec.canonical_unit,
        matched=matched,
        missing_data=missing,
    ), expression


def _compare(actual: Any, operator: Operator, expected: Any) -> bool:
    if operator == Operator.EQ:
        return actual == expected
    if operator == Operator.NEQ:
        return actual != expected
    if operator == Operator.GT:
        return actual > expected
    if operator == Operator.GTE:
        return actual >= expected
    if operator == Operator.LT:
        return actual < expected
    if operator == Operator.LTE:
        return actual <= expected
    if operator == Operator.IN:
        return actual in expected
    if operator == Operator.NOT_IN:
        return actual not in expected
    raise ValueError(f"Unsupported operator {operator!r}")


def _evaluate_conditions(conditions: Conditions, weather: NormalizedWeather) -> tuple[bool, tuple[ConditionEvidence, ...], tuple[str, ...]]:
    evidence: list[ConditionEvidence] = []
    matched_expressions: list[str] = []
    all_matched = True
    any_matched = True

    if conditions.all is not None:
        all_results = [_condition_evidence("all", condition, weather) for condition in conditions.all]
        evidence.extend(item[0] for item in all_results)
        matched_expressions.extend(item[1] for item in all_results if item[0].matched)
        all_matched = all(item[0].matched for item in all_results)

    if conditions.any is not None:
        any_results = [_condition_evidence("any", condition, weather) for condition in conditions.any]
        evidence.extend(item[0] for item in any_results)
        matched_expressions.extend(item[1] for item in any_results if item[0].matched)
        any_matched = any(item[0].matched for item in any_results)

    return all_matched and any_matched, tuple(evidence), tuple(matched_expressions)


def _policy_is_applicable(policy: Policy, request: PolicyRequest, on_date: date) -> bool:
    return (
        request.activity in policy.activities
        and (not policy.audiences or request.audience in policy.audiences)
        and policy.effective_from <= on_date
        and (policy.effective_until is None or on_date <= policy.effective_until)
    )


def evaluate_policies(
    request: PolicyRequest,
    weather: NormalizedWeather,
    policies: PolicySet | tuple[Policy, ...] | list[Policy],
    *,
    as_of: date | None = None,
) -> PolicyEvaluationResult:
    """Evaluate every eligible SOP and return all matches with condition evidence.

    Matches sort by severity descending, priority descending, then policy_id ascending.
    Lower-severity matches are retained. Explicit unresolved conflict declarations
    change the result status to POLICY_CONFLICT without discarding either match.
    """
    evaluation_date = as_of or date.today()
    policy_items = policies.policies if isinstance(policies, PolicySet) else policies
    matches: list[PolicyMatch] = []

    for policy in policy_items:
        if not _policy_is_applicable(policy, request, evaluation_date):
            continue
        matched, condition_results, matched_conditions = _evaluate_conditions(policy.conditions, weather)
        if not matched:
            continue
        matches.append(
            PolicyMatch(
                policy_id=policy.policy_id,
                version=policy.version,
                severity=policy.severity,
                priority=policy.priority,
                matched_conditions=matched_conditions,
                condition_results=condition_results,
                relevant_weather_fields=tuple(dict.fromkeys(item.field for item in condition_results)),
                directive=policy.directive,
            )
        )

    matches.sort(
        key=lambda match: (
            -SEVERITY_RANK[match.severity],
            -match.priority,
            match.policy_id,
        )
    )
    if not matches:
        return PolicyEvaluationResult(status=EvaluationStatus.NO_MATCH)

    matched_ids = {match.policy_id for match in matches}
    policies_by_id = {policy.policy_id: policy for policy in policy_items}
    conflict_pairs: set[tuple[str, str]] = set()
    for policy_id in matched_ids:
        for other_id in policies_by_id[policy_id].conflicts_with:
            if other_id in matched_ids:
                conflict_pairs.add(tuple(sorted((policy_id, other_id))))
    conflicts = tuple(
        PolicyConflict(
            policy_ids=pair,
            reason="Both explicitly conflicting SOPs apply and no resolution rule is configured.",
        )
        for pair in sorted(conflict_pairs)
    )
    status = EvaluationStatus.POLICY_CONFLICT if conflicts else EvaluationStatus.MATCH
    return PolicyEvaluationResult(status=status, matches=tuple(matches), conflicts=conflicts)