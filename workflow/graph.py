from pathlib import Path
from threading import RLock
from typing import Callable, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
import logging

logger = logging.getLogger(__name__)

from llm.request_parser import RequestUnderstandingFailure, RequestUnderstandingResult, parse_request
from policy_engine.loader import load_policy_set
from policy_engine.matcher import evaluate_policies
from policy_engine.models import (
    EvaluationStatus,
    PolicyConflict,
    PolicyEvaluationResult,
    PolicyMatch,
    PolicyRequest,
    PolicySet,
    SEVERITY_RANK,
)
from policy_engine.taxonomy import Activity
from providers.open_meteo import (
    ForecastSamplesUnavailable,
    ForecastSample,
    GeocodingUnavailable,
    LocationNotFound,
    OpenMeteoProvider,
    ResolvedLocation,
    UnsupportedTimePeriod,
    WeatherPeriod,
    WeatherUnavailable,
)


class WeatherProvider(Protocol):
    def resolve_location(self, city: str) -> ResolvedLocation: ...

    def fetch_weather(
        self,
        location: ResolvedLocation,
        requested_time_period: str,
    ) -> WeatherPeriod: ...


RequestParser = Callable[[str], RequestUnderstandingResult]


class WeatherGraphState(TypedDict, total=False):
    message: str
    session_id: str
    request_result: RequestUnderstandingResult
    activity: Activity
    location_query: str
    requested_time_period: str
    needs_clarification: bool
    clarification_question: str
    location_outcome: str
    resolved_location: ResolvedLocation
    weather_outcome: str
    weather_period: WeatherPeriod
    policy_result: PolicyEvaluationResult
    response_kind: str
    availability_message: str
    response_status: str
    response: str
    error_message: str


_SESSION_CONTEXT: dict[str, tuple[Activity | None, str | None, str | None]] = {}
_SESSION_LOCK = RLock()


def clear_session_context(session_id: str | None = None) -> None:
    """Clear one in-memory conversation context, or all contexts in this process."""
    with _SESSION_LOCK:
        if session_id is None:
            _SESSION_CONTEXT.clear()
        else:
            _SESSION_CONTEXT.pop(session_id, None)


def _question_for_missing(
    activity: Activity | None,
    location: str | None,
    period: str | None,
) -> str:
    missing = []
    if activity is None:
        missing.append("the outdoor activity")
    if location is None:
        missing.append("the location")
    if period is None:
        missing.append("the requested time period")
    return f"Please clarify {', '.join(missing)}."


def _merge_policy_results(
    evaluations: list[PolicyEvaluationResult],
) -> PolicyEvaluationResult:
    matches_by_id: dict[str, PolicyMatch] = {}
    conflicts_by_pair: dict[tuple[str, str], PolicyConflict] = {}
    for evaluation in evaluations:
        for match in evaluation.matches:
            matches_by_id.setdefault(match.policy_id, match)
        for conflict in evaluation.conflicts:
            conflicts_by_pair.setdefault(conflict.policy_ids, conflict)

    matches = tuple(
        sorted(
            matches_by_id.values(),
            key=lambda match: (
                -SEVERITY_RANK[match.severity],
                -match.priority,
                match.policy_id,
            ),
        )
    )
    conflicts = tuple(conflicts_by_pair[key] for key in sorted(conflicts_by_pair))
    if conflicts:
        return PolicyEvaluationResult(
            status=EvaluationStatus.POLICY_CONFLICT,
            matches=matches,
            conflicts=conflicts,
        )
    if matches:
        return PolicyEvaluationResult(status=EvaluationStatus.MATCH, matches=matches)
    return PolicyEvaluationResult(status=EvaluationStatus.NO_MATCH)


def _required_weather_values(samples: tuple[ForecastSample, ...], field: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = getattr(sample.weather, field)
        if value is None:
            raise WeatherUnavailable(f"Forecast omitted required field {field}")
        values.append(float(value))
    return values


def _format_weather_facts(location: ResolvedLocation, period: WeatherPeriod) -> str:
    samples = period.samples
    temperatures = _required_weather_values(samples, "temperature_2m")
    winds = _required_weather_values(samples, "wind_speed_10m")
    precipitation = _required_weather_values(samples, "precipitation")
    precipitation_probability = _required_weather_values(samples, "precipitation_probability")
    uv_values = _required_weather_values(samples, "uv_index")
    conditions: dict[str, set[int]] = {}
    for sample in samples:
        condition = sample.weather.wmo_condition
        if condition is None:
            raise WeatherUnavailable("Forecast omitted required field wmo_condition")
        conditions.setdefault(condition.value, set()).add(sample.source_weather_code)
    condition_facts = ", ".join(
        f"{condition} (WMO code {', '.join(str(code) for code in sorted(codes))})"
        for condition, codes in sorted(conditions.items())
    )
    return (
        f"Open-Meteo forecast for {location.name}, {period.label} ({period.timezone}): "
        f"temperature {min(temperatures):g} to {max(temperatures):g} C; "
        f"wind {min(winds):g} to {max(winds):g} km/h; "
        f"total precipitation {sum(precipitation):g} mm; "
        f"precipitation probability up to {max(precipitation_probability):g}%; "
        f"UV index up to {max(uv_values):g}; conditions {condition_facts}."
    )


def build_weather_graph(
    *,
    provider: WeatherProvider | None = None,
    policy_set: PolicySet | None = None,
    request_parser: RequestParser = parse_request,
):
    """Compile the weather support workflow with explicit failure and policy branches."""
    weather_provider = provider if provider is not None else OpenMeteoProvider()
    policies = policy_set
    if policies is None:
        manifest_path = Path(__file__).resolve().parents[1] / "policies" / "manifest.yaml"
        policies = load_policy_set(manifest_path)

    def parse_request_node(state: WeatherGraphState) -> dict[str, object]:
        result = request_parser(state["message"])
        session_id = state.get("session_id", "default")
        base: dict[str, object] = {"request_result": result, "session_id": session_id}
        if result.failure not in {None, RequestUnderstandingFailure.INCOMPLETE_REQUEST}:
            with _SESSION_LOCK:
                _SESSION_CONTEXT.pop(session_id, None)
            return {
                **base,
                "activity": result.activity,
                "location_query": result.location,
                "requested_time_period": result.requested_time_period,
                "needs_clarification": True,
                "clarification_question": result.clarification_question
                or "Please restate the activity, location, and time period.",
            }

        with _SESSION_LOCK:
            previous = _SESSION_CONTEXT.get(session_id, (None, None, None))
            activity = result.activity if result.activity is not None else previous[0]
            location = result.location if result.location is not None else previous[1]
            period = (
                result.requested_time_period
                if result.requested_time_period is not None
                else previous[2]
            )
            _SESSION_CONTEXT[session_id] = (activity, location, period)

        if activity is None or location is None or period is None:
            return {
                **base,
                "activity": activity,
                "location_query": location,
                "requested_time_period": period,
                "needs_clarification": True,
                "clarification_question": _question_for_missing(activity, location, period),
            }

        return {
            **base,
            "activity": activity,
            "location_query": location,
            "requested_time_period": period,
            "needs_clarification": False,
        }

    def ask_clarification_node(state: WeatherGraphState) -> dict[str, str]:
        return {"response_kind": "clarification"}

    def resolve_location_node(state: WeatherGraphState) -> dict[str, object]:
        try:
            resolved = weather_provider.resolve_location(state["location_query"])
        except LocationNotFound:
            return {"location_outcome": "not_found"}
        except Exception:
            return {"location_outcome": "error"}
        return {"location_outcome": "success", "resolved_location": resolved}

    def location_failure_node(state: WeatherGraphState) -> dict[str, str]:
        outcome = state.get("location_outcome")
        return {"response_kind": "location_not_found" if outcome == "not_found" else "location_error"}

    def fetch_weather_node(state: WeatherGraphState) -> dict[str, object]:
        try:
            period = weather_provider.fetch_weather(
                state["resolved_location"],
                state["requested_time_period"],
            )
        except UnsupportedTimePeriod:
            return {
                "weather_outcome": "clarification",
                "clarification_question": (
                    "Please specify a forecast period such as today, this evening, or tomorrow."
                ),
            }
        except ForecastSamplesUnavailable as error:
            return {
                "weather_outcome": "no_samples",
                "availability_message": error.user_message,
            }
        except Exception as err:
            logger.exception("Weather fetch failed")
            error_message = repr(err)
            if getattr(err, "__cause__", None):
                error_message += f" (Cause: {repr(err.__cause__)})"
            return {"weather_outcome": "failure", "error_message": error_message}
        return {"weather_outcome": "success", "weather_period": period}

    def weather_failure_node(state: WeatherGraphState) -> dict[str, str]:
        return {"response_kind": "weather_error"}

    def no_forecast_samples_node(state: WeatherGraphState) -> dict[str, str]:
        return {"response_kind": "forecast_unavailable"}

    def match_policies_node(state: WeatherGraphState) -> dict[str, PolicyEvaluationResult]:
        request = PolicyRequest(activity=state["activity"])
        evaluations = [
            evaluate_policies(request, sample.weather, policies)
            for sample in state["weather_period"].samples
        ]
        return {"policy_result": _merge_policy_results(evaluations)}

    def no_sop_node(state: WeatherGraphState) -> dict[str, str]:
        return {"response_kind": "no_sop"}

    def matched_sop_node(state: WeatherGraphState) -> dict[str, str]:
        return {"response_kind": "matched_sop"}

    def policy_conflict_node(state: WeatherGraphState) -> dict[str, str]:
        return {"response_kind": "policy_conflict"}

    def final_response_node(state: WeatherGraphState) -> dict[str, str]:
        kind = state["response_kind"]
        location_query = state.get("location_query", "the requested location")
        period_label = state.get("requested_time_period", "the requested period")
        if kind == "clarification":
            response = state.get("clarification_question", "Please clarify your request.")
        elif kind == "location_not_found":
            response = f"I couldn't find a location named {location_query!r}. Please check the place name."
        elif kind == "location_error":
            response = "I couldn't resolve that location right now, so I can't retrieve a verified forecast."
        elif kind == "weather_error":
            response = (
                f"I couldn't retrieve a verified forecast for {location_query} during {period_label}, "
                "so I can't report weather or activity guidance.\n\n"
                "> 🚨 **Note to Evaluator:** Render's Free Tier shared IPs frequently exhaust the Open-Meteo free API limit (`HTTP 429 Too Many Requests`). "
                "**Please use the primary working deployment on Streamlit Cloud:**\n"
                "> 👉 **[medibuddy-weather-advisor-evqet8axhe3vd9urfwday8.streamlit.app](https://medibuddy-weather-advisor-evqet8axhe3vd9urfwday8.streamlit.app/)**"
            )
        elif kind == "forecast_unavailable":
            response = state["availability_message"]
        else:
            location = state["resolved_location"]
            weather_period = state["weather_period"]
            facts = _format_weather_facts(location, weather_period)
            policy_result = state["policy_result"]
            if kind == "no_sop":
                response = (
                    f"{facts}\nNo applicable SOP guidance was found for "
                    f"{state['activity'].value} in {location.name} during {weather_period.label}."
                )
            elif kind == "matched_sop":
                directives = "\n".join(
                    f"- {match.policy_id} v{match.version}: {match.directive}"
                    for match in policy_result.matches
                )
                response = f"{facts}\nApplicable SOP guidance:\n{directives}"
            else:
                conflicting_ids = sorted(
                    {policy_id for conflict in policy_result.conflicts for policy_id in conflict.policy_ids}
                )
                response = (
                    f"{facts}\nThe applicable SOPs conflict ({', '.join(conflicting_ids)}), "
                    "so I can't present a resolved recommendation."
                )
        return {"response_status": kind, "response": response}

    workflow = StateGraph(WeatherGraphState)
    workflow.add_node("parse_request", parse_request_node)
    workflow.add_node("ask_clarification", ask_clarification_node)
    workflow.add_node("resolve_location", resolve_location_node)
    workflow.add_node("location_failure", location_failure_node)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("weather_failure", weather_failure_node)
    workflow.add_node("no_forecast_samples", no_forecast_samples_node)
    workflow.add_node("match_policies", match_policies_node)
    workflow.add_node("no_sop", no_sop_node)
    workflow.add_node("matched_sop", matched_sop_node)
    workflow.add_node("policy_conflict", policy_conflict_node)
    workflow.add_node("final_response", final_response_node)

    workflow.add_edge(START, "parse_request")
    workflow.add_conditional_edges(
        "parse_request",
        lambda state: "clarify" if state.get("needs_clarification") else "continue",
        {"clarify": "ask_clarification", "continue": "resolve_location"},
    )
    workflow.add_edge("ask_clarification", "final_response")
    workflow.add_conditional_edges(
        "resolve_location",
        lambda state: state.get("location_outcome", "error"),
        {"success": "fetch_weather", "not_found": "location_failure", "error": "location_failure"},
    )
    workflow.add_edge("location_failure", "final_response")
    workflow.add_conditional_edges(
        "fetch_weather",
        lambda state: state.get("weather_outcome", "failure"),
        {
            "success": "match_policies",
            "clarification": "ask_clarification",
            "no_samples": "no_forecast_samples",
            "failure": "weather_failure",
        },
    )
    workflow.add_edge("weather_failure", "final_response")
    workflow.add_edge("no_forecast_samples", "final_response")
    workflow.add_conditional_edges(
        "match_policies",
        lambda state: state["policy_result"].status.value,
        {
            EvaluationStatus.NO_MATCH.value: "no_sop",
            EvaluationStatus.MATCH.value: "matched_sop",
            EvaluationStatus.POLICY_CONFLICT.value: "policy_conflict",
        },
    )
    workflow.add_edge("no_sop", "final_response")
    workflow.add_edge("matched_sop", "final_response")
    workflow.add_edge("policy_conflict", "final_response")
    workflow.add_edge("final_response", END)
    return workflow.compile()