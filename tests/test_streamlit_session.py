from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from llm.request_parser import RequestUnderstandingFailure, RequestUnderstandingResult
from policy_engine.taxonomy import Activity
from policy_engine.weather import NormalizedWeather, WeatherCondition
from providers.open_meteo import ForecastSample, ResolvedLocation, WeatherPeriod
from workflow import graph as graph_module
from workflow.graph import clear_session_context


ROOT = Path(__file__).parents[1]


def test_streamlit_two_turn_chat_reuses_location_in_same_session(monkeypatch):
    provider_calls = []

    class RecordingProvider:
        def resolve_location(self, city):
            provider_calls.append(("resolve", city))
            return ResolvedLocation(name=city, latitude=23.2, longitude=77.4)

        def fetch_weather(self, location, requested_time_period):
            provider_calls.append(("fetch", location.name, requested_time_period))
            return WeatherPeriod(
                label=requested_time_period,
                timezone="Asia/Kolkata",
                samples=(
                    ForecastSample(
                        time=datetime.now(timezone.utc),
                        weather=NormalizedWeather(
                            temperature_2m=24,
                            wind_speed_10m=6,
                            precipitation=0,
                            precipitation_probability=0,
                            uv_index=0,
                            wmo_condition=WeatherCondition.CLEAR,
                        ),
                        source_weather_code=0,
                    ),
                ),
            )

    def request_parser(message):
        if message == "Can I cycle in Bhopal today?":
            return RequestUnderstandingResult(
                activity=Activity.CYCLING,
                location="Bhopal",
                requested_time_period="today",
                clarification_needed=False,
            )
        assert message == "Can I go cycling this evening?"
        return RequestUnderstandingResult(
            activity=Activity.CYCLING,
            location=None,
            requested_time_period="evening",
            clarification_needed=True,
            failure=RequestUnderstandingFailure.INCOMPLETE_REQUEST,
        )

    original_builder = graph_module.build_weather_graph

    def build_test_graph(**kwargs):
        return original_builder(
            provider=RecordingProvider(),
            request_parser=request_parser,
            **kwargs,
        )

    monkeypatch.setattr(graph_module, "build_weather_graph", build_test_graph)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    session_id = app.session_state["session_id"]

    app.chat_input[0].set_value("Can I cycle in Bhopal today?").run()
    assert not app.exception
    assert app.session_state["session_id"] == session_id

    app.chat_input[0].set_value("Can I go cycling this evening?").run()
    assert not app.exception
    assert app.session_state["session_id"] == session_id
    assert ("fetch", "Bhopal", "today") in provider_calls
    assert ("fetch", "Bhopal", "evening") in provider_calls
    assert not any(call[0] == "resolve" and call[1] is None for call in provider_calls)

    clear_session_context(session_id)


def test_streamlit_three_turn_clarification_merges_partial_slots(monkeypatch):
    provider_calls = []

    class RecordingProvider:
        def resolve_location(self, city):
            provider_calls.append(("resolve", city))
            return ResolvedLocation(name=city, latitude=23.2, longitude=77.4)

        def fetch_weather(self, location, requested_time_period):
            provider_calls.append(("fetch", location.name, requested_time_period))
            return WeatherPeriod(
                label=requested_time_period,
                timezone="Asia/Kolkata",
                samples=(
                    ForecastSample(
                        time=datetime.now(timezone.utc),
                        weather=NormalizedWeather(
                            temperature_2m=24,
                            wind_speed_10m=6,
                            precipitation=0,
                            precipitation_probability=0,
                            uv_index=0,
                            wmo_condition=WeatherCondition.CLEAR,
                        ),
                        source_weather_code=0,
                    ),
                ),
            )

    first_request = RequestUnderstandingResult(
        activity=Activity.CYCLING,
        location=None,
        requested_time_period="evening",
        clarification_needed=True,
        failure=RequestUnderstandingFailure.INCOMPLETE_REQUEST,
    )
    location_clarification = RequestUnderstandingResult(
        activity=None,
        location="Bhopal",
        requested_time_period=None,
        clarification_needed=True,
        failure=RequestUnderstandingFailure.INCOMPLETE_REQUEST,
    )
    activity_clarification = RequestUnderstandingResult(
        activity=Activity.CYCLING,
        location=None,
        requested_time_period=None,
        clarification_needed=True,
        failure=RequestUnderstandingFailure.INCOMPLETE_REQUEST,
    )
    parser_results = {
        "Can I go cycling this evening?": first_request,
        "Bhopal": location_clarification,
        "cycling": activity_clarification,
    }

    def request_parser(message):
        return parser_results[message]

    original_builder = graph_module.build_weather_graph

    def build_test_graph(**kwargs):
        return original_builder(
            provider=RecordingProvider(),
            request_parser=request_parser,
            **kwargs,
        )

    monkeypatch.setattr(graph_module, "build_weather_graph", build_test_graph)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    session_id = app.session_state["session_id"]

    app.chat_input[0].set_value("Can I go cycling this evening?").run()
    assert not app.exception
    assert any("Please clarify the location." in element.value for element in app.info)
    assert provider_calls == []

    app.chat_input[0].set_value("Bhopal").run()
    assert not app.exception
    assert app.session_state["session_id"] == session_id
    assert ("fetch", "Bhopal", "evening") in provider_calls
    assert not any(
        "Please clarify the outdoor activity" in element.value
        or "Please clarify the requested time period" in element.value
        for element in app.info
    )

    calls_before_third_turn = len(provider_calls)
    app.chat_input[0].set_value("cycling").run()
    assert not app.exception
    assert app.session_state["session_id"] == session_id
    assert len(provider_calls) > calls_before_third_turn
    assert provider_calls[-2:] == [
        ("resolve", "Bhopal"),
        ("fetch", "Bhopal", "evening"),
    ]

    clear_session_context(session_id)


@pytest.mark.parametrize(
    "parser_failure",
    [
        RequestUnderstandingFailure.UNSUPPORTED_ACTIVITY,
        RequestUnderstandingFailure.OPENAI_ERROR,
    ],
)
def test_streamlit_parser_failure_clears_stale_intent_before_city_follow_up(
    monkeypatch,
    parser_failure,
):
    provider_calls = []

    class RecordingProvider:
        def resolve_location(self, city):
            provider_calls.append(("resolve", city))
            return ResolvedLocation(name=city, latitude=26.9, longitude=75.8)

        def fetch_weather(self, location, requested_time_period):
            provider_calls.append(("fetch", location.name, requested_time_period))
            return WeatherPeriod(
                label=requested_time_period,
                timezone="Asia/Kolkata",
                samples=(
                    ForecastSample(
                        time=datetime.now(timezone.utc),
                        weather=NormalizedWeather(
                            temperature_2m=24,
                            wind_speed_10m=6,
                            precipitation=0,
                            precipitation_probability=0,
                            uv_index=0,
                            wmo_condition=WeatherCondition.CLEAR,
                        ),
                        source_weather_code=0,
                    ),
                ),
            )

    parser_results = {
        "Can I cycle in Bhopal today?": RequestUnderstandingResult(
            activity=Activity.CYCLING,
            location="Bhopal",
            requested_time_period="today",
            clarification_needed=False,
        ),
        "new request with unsupported activity": RequestUnderstandingResult(
            activity=None,
            location="Jaipur" if parser_failure == RequestUnderstandingFailure.UNSUPPORTED_ACTIVITY else None,
            requested_time_period="tomorrow" if parser_failure == RequestUnderstandingFailure.UNSUPPORTED_ACTIVITY else None,
            clarification_needed=True,
            clarification_question="Which supported outdoor activity do you mean?",
            failure=parser_failure,
        ),
        "Jaipur": RequestUnderstandingResult(
            activity=None,
            location="Jaipur",
            requested_time_period=None,
            clarification_needed=True,
            failure=RequestUnderstandingFailure.INCOMPLETE_REQUEST,
        ),
        "Can I cycle in Jaipur tomorrow?": RequestUnderstandingResult(
            activity=Activity.CYCLING,
            location="Jaipur",
            requested_time_period="tomorrow",
            clarification_needed=False,
        ),
    }

    def request_parser(message):
        return parser_results[message]

    original_builder = graph_module.build_weather_graph

    def build_test_graph(**kwargs):
        return original_builder(
            provider=RecordingProvider(),
            request_parser=request_parser,
            **kwargs,
        )

    monkeypatch.setattr(graph_module, "build_weather_graph", build_test_graph)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    session_id = app.session_state["session_id"]

    app.chat_input[0].set_value("Can I cycle in Bhopal today?").run()
    assert not app.exception
    assert ("fetch", "Bhopal", "today") in provider_calls

    calls_before_failure = len(provider_calls)
    app.chat_input[0].set_value("new request with unsupported activity").run()
    assert not app.exception
    assert len(provider_calls) == calls_before_failure

    app.chat_input[0].set_value("Jaipur").run()
    assert not app.exception
    assert app.session_state["session_id"] == session_id
    assert len(provider_calls) == calls_before_failure
    assert any(
        "Please clarify the outdoor activity" in element.value
        and "requested time period" in element.value
        for element in app.info
    )

    app.chat_input[0].set_value("Can I cycle in Jaipur tomorrow?").run()
    assert not app.exception
    assert provider_calls[-2:] == [
        ("resolve", "Jaipur"),
        ("fetch", "Jaipur", "tomorrow"),
    ]

    clear_session_context(session_id)