import re
from datetime import date
from enum import StrEnum
from typing import TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from policy_engine.fields import FieldKind, WEATHER_FIELD_SPECS
from policy_engine.taxonomy import Activity, Audience, PolicyCategory
from policy_engine.weather import WeatherCondition


_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
ConditionScalar: TypeAlias = StrictStr | StrictInt | StrictFloat
ConditionValue: TypeAlias = ConditionScalar | list[ConditionScalar]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Severity(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MODERATE: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Operator(StrEnum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"


class Condition(StrictModel):
    field: str
    operator: Operator
    value: ConditionValue
    unit: str | None = None

    @model_validator(mode="after")
    def validate_field_operator_and_value(self) -> "Condition":
        spec = WEATHER_FIELD_SPECS.get(self.field)
        if spec is None:
            allowed = ", ".join(WEATHER_FIELD_SPECS)
            raise ValueError(f"Unknown weather field {self.field!r}; allowed fields: {allowed}")

        values = self.value if isinstance(self.value, list) else [self.value]
        if not values:
            raise ValueError("Condition value lists must not be empty")
        if self.operator in {Operator.IN, Operator.NOT_IN} and not isinstance(self.value, list):
            raise ValueError(f"Operator {self.operator.value!r} requires a list value")
        if self.operator not in {Operator.IN, Operator.NOT_IN} and isinstance(self.value, list):
            raise ValueError(f"Operator {self.operator.value!r} requires a scalar value")

        if spec.kind == FieldKind.NUMERIC:
            if self.unit != spec.canonical_unit:
                raise ValueError(
                    f"Field {self.field!r} requires canonical unit {spec.canonical_unit!r}"
                )
            if any(isinstance(value, str) for value in values):
                raise ValueError(f"Field {self.field!r} requires numeric condition values")
            if self.operator in {Operator.GT, Operator.GTE, Operator.LT, Operator.LTE} and isinstance(
                self.value, list
            ):
                raise ValueError("Ordered comparisons require a scalar value")
        else:
            if self.unit is not None:
                raise ValueError(f"Categorical field {self.field!r} must not declare a unit")
            if self.operator in {Operator.GT, Operator.GTE, Operator.LT, Operator.LTE}:
                raise ValueError(f"Ordered comparison {self.operator.value!r} is not valid for categories")
            if any(not isinstance(value, str) for value in values):
                raise ValueError(f"Field {self.field!r} requires categorical string values")
            allowed_values = spec.allowed_values
            invalid_values = sorted(set(values) - allowed_values)
            if invalid_values:
                raise ValueError(
                    f"Invalid value(s) for {self.field!r}: {invalid_values}; "
                    f"allowed values: {sorted(allowed_values)}"
                )
        return self


class Conditions(StrictModel):
    all: tuple[Condition, ...] | None = None
    any: tuple[Condition, ...] | None = None

    @model_validator(mode="after")
    def require_nonempty_group(self) -> "Conditions":
        if self.all is None and self.any is None:
            raise ValueError("Conditions must define a non-empty 'all' group, an 'any' group, or both")
        if self.all == () or self.any == ():
            raise ValueError("Condition groups must not be empty")
        return self


class Policy(StrictModel):
    policy_id: StrictStr = Field(min_length=1, pattern=r"^[A-Z][A-Z0-9_-]*$")
    version: StrictStr
    category: PolicyCategory
    activities: tuple[Activity, ...] = Field(min_length=1)
    audiences: tuple[Audience, ...] = ()
    severity: Severity
    priority: StrictInt = Field(ge=0)
    effective_from: date
    effective_until: date | None
    conditions: Conditions
    directive: StrictStr = Field(min_length=1)
    rationale: StrictStr = Field(min_length=1)
    source: StrictStr = Field(min_length=1)
    conflicts_with: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def validate_policy(self) -> "Policy":
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"Invalid policy version {self.version!r}; expected semantic versioning")
        if self.effective_until is not None and self.effective_until < self.effective_from:
            raise ValueError("effective_until must be on or after effective_from")
        if len(set(self.activities)) != len(self.activities):
            raise ValueError("activities must not contain duplicates")
        if len(set(self.audiences)) != len(self.audiences):
            raise ValueError("audiences must not contain duplicates")
        if len(set(self.conflicts_with)) != len(self.conflicts_with):
            raise ValueError("conflicts_with must not contain duplicates")
        if self.policy_id in self.conflicts_with:
            raise ValueError("A policy cannot conflict with itself")
        return self


class PolicyManifest(StrictModel):
    policy_set_version: StrictStr
    updated_at: date
    description: StrictStr = Field(min_length=1)
    policies_file: StrictStr = Field(pattern=r"^[^/\\]+\.ya?ml$")

    @model_validator(mode="after")
    def validate_version(self) -> "PolicyManifest":
        if not _VERSION_PATTERN.fullmatch(self.policy_set_version):
            raise ValueError(
                f"Invalid policy-set version {self.policy_set_version!r}; expected semantic versioning"
            )
        return self


class PolicyDocument(StrictModel):
    policy_set_version: StrictStr
    sops: tuple[Policy, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_document(self) -> "PolicyDocument":
        if not _VERSION_PATTERN.fullmatch(self.policy_set_version):
            raise ValueError(
                f"Invalid policy-set version {self.policy_set_version!r}; expected semantic versioning"
            )
        _validate_policy_collection(self.sops)
        return self


class PolicySet(StrictModel):
    manifest: PolicyManifest
    policy_set_version: StrictStr
    policies: tuple[Policy, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def match_manifest_version(self) -> "PolicySet":
        if not _VERSION_PATTERN.fullmatch(self.policy_set_version):
            raise ValueError("Invalid policy-set version; expected semantic versioning")
        if self.manifest.policy_set_version != self.policy_set_version:
            raise ValueError("Policy-set version must match the manifest version")
        _validate_policy_collection(self.policies)
        return self


def _validate_policy_collection(policies: tuple[Policy, ...]) -> None:
    identifiers = [policy.policy_id for policy in policies]
    duplicates = sorted({policy_id for policy_id in identifiers if identifiers.count(policy_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate policy_id value(s): {', '.join(duplicates)}")
    policies_by_id = {policy.policy_id: policy for policy in policies}
    for policy in policies:
        for conflicting_id in policy.conflicts_with:
            conflicting = policies_by_id.get(conflicting_id)
            if conflicting is None:
                raise ValueError(
                    f"Policy {policy.policy_id!r} conflicts with unknown policy {conflicting_id!r}"
                )
            if policy.policy_id not in conflicting.conflicts_with:
                raise ValueError(
                    f"Conflict declaration between {policy.policy_id!r} and "
                    f"{conflicting_id!r} must be reciprocal"
                )


class PolicyRequest(StrictModel):
    activity: Activity
    audience: Audience | None = None


class ConditionEvidence(StrictModel):
    group: str
    field: str
    operator: Operator
    threshold: ConditionValue
    actual: ConditionScalar | None
    unit: str | None
    matched: bool
    missing_data: bool = False


class PolicyMatch(StrictModel):
    policy_id: str
    version: str
    severity: Severity
    priority: int
    matched_conditions: tuple[str, ...]
    condition_results: tuple[ConditionEvidence, ...]
    relevant_weather_fields: tuple[str, ...]
    directive: str


class PolicyConflict(StrictModel):
    policy_ids: tuple[str, str]
    reason: str


class EvaluationStatus(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    POLICY_CONFLICT = "POLICY_CONFLICT"


class PolicyEvaluationResult(StrictModel):
    status: EvaluationStatus
    matches: tuple[PolicyMatch, ...] = ()
    conflicts: tuple[PolicyConflict, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> "PolicyEvaluationResult":
        if self.status == EvaluationStatus.NO_MATCH and (self.matches or self.conflicts):
            raise ValueError("NO_MATCH results cannot contain matches or conflicts")
        if self.status == EvaluationStatus.MATCH and (not self.matches or self.conflicts):
            raise ValueError("MATCH results require matches and cannot contain conflicts")
        if self.status == EvaluationStatus.POLICY_CONFLICT and (not self.matches or not self.conflicts):
            raise ValueError("POLICY_CONFLICT results require matches and at least one conflict")
        return self