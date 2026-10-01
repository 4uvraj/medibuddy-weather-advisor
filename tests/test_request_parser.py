from types import SimpleNamespace

import pytest

from llm.request_parser import (
    RequestExtraction,
    RequestUnderstandingFailure,
    parse_request,
)
from policy_engine.taxonomy import Activity


class FakeResponses:
    def __init__(self, parsed=None, error=None):
        self.parsed = parsed
        self.error = error
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed=None, error=None):
        self.responses = FakeResponses(parsed=parsed, error=error)


def extraction(activity="bike ride", location="Bhopal", period="today"):
    return RequestExtraction(
        activity=activity,
        location=location,
        requested_time_period=period,
    )


def test_responses_api_extracts_slots_and_normalizes_activity():
    client = FakeClient(parsed=extraction())

    result = parse_request("Can I take a bike ride in Bhopal today?", client=client, model="test-model")

    assert result.clarification_needed is False
    assert result.activity == Activity.CYCLING
    assert result.location == "Bhopal"
    assert result.requested_time_period == "today"
    assert client.responses.call["model"] == "test-model"
    assert client.responses.call["text_format"] is RequestExtraction
    assert client.responses.call["input"][1] == {
        "role": "user",
        "content": "Can I take a bike ride in Bhopal today?",
    }


@pytest.mark.parametrize(
    "activity_variant",
    ["cycle", "cycling", "bike", "biking", "bike ride", "cycle ride", "ride a bike"],
)
def test_common_cycling_variants_normalize_to_canonical_activity(activity_variant):
    client = FakeClient(parsed=extraction(activity=activity_variant))

    result = parse_request(
        "Is it safe to cycle in Bhopal today?",
        client=client,
        model="test-model",
    )

    assert result.activity == Activity.CYCLING


def test_exact_live_query_slots_normalize_extracted_cycle_variant():
    client = FakeClient(parsed=extraction(activity="cycle", location="Bhopal", period="today"))

    result = parse_request(
        "Is it safe to cycle in Bhopal today?",
        client=client,
        model="test-model",
    )

    assert result.clarification_needed is False
    assert result.activity == Activity.CYCLING
    assert result.location == "Bhopal"
    assert result.requested_time_period == "today"


def test_request_extraction_schema_contains_only_the_three_allowed_slots():
    schema = RequestExtraction.model_json_schema()

    assert set(schema["properties"]) == {"activity", "location", "requested_time_period"}
    assert set(schema["required"]) == {"activity", "location", "requested_time_period"}
    assert schema["additionalProperties"] is False


def test_unknown_activity_is_rejected_instead_of_passed_to_policy_engine():
    client = FakeClient(parsed=extraction(activity="unlisted extreme sport"))

    result = parse_request("Do my extreme sport?", client=client, model="test-model")

    assert result.clarification_needed is True
    assert result.activity is None
    assert result.location == "Bhopal"
    assert result.failure == RequestUnderstandingFailure.UNSUPPORTED_ACTIVITY
    assert "Which supported outdoor activity" in result.clarification_question


def test_missing_extraction_values_lead_to_targeted_clarification():
    client = FakeClient(parsed=extraction(activity=None, location=" ", period=None))

    result = parse_request("Is it safe outside?", client=client, model="test-model")

    assert result.clarification_needed is True
    assert result.failure == RequestUnderstandingFailure.INCOMPLETE_REQUEST
    assert "outdoor activity" in result.clarification_question
    assert "location" in result.clarification_question
    assert "time period" in result.clarification_question


def test_openai_failure_returns_clarification_without_guessed_values():
    client = FakeClient(error=RuntimeError("network unavailable"))

    result = parse_request("Is cycling okay in Bhopal this evening?", client=client, model="test-model")

    assert result.clarification_needed is True
    assert result.failure == RequestUnderstandingFailure.OPENAI_ERROR
    assert result.activity is None
    assert result.location is None
    assert result.requested_time_period is None
    assert result.clarification_question is not None


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_message_is_handled_without_openai_call(message):
    client = FakeClient(parsed=extraction())

    result = parse_request(message, client=client, model="test-model")

    assert result.failure == RequestUnderstandingFailure.EMPTY_MESSAGE
    assert result.clarification_needed is True
    assert client.responses.call is None


def test_missing_model_configuration_returns_clarification(monkeypatch):
    monkeypatch.setattr("llm.request_parser.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = FakeClient(parsed=extraction())

    result = parse_request("Can I cycle in Bhopal today?", client=client)

    assert result.failure == RequestUnderstandingFailure.MISSING_CONFIGURATION
    assert result.clarification_needed is True
    assert client.responses.call is None


def test_missing_api_key_returns_clarification_without_openai_call(monkeypatch):
    monkeypatch.setattr("llm.request_parser.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = parse_request("Can I cycle in Bhopal today?")

    assert result.failure == RequestUnderstandingFailure.MISSING_CONFIGURATION
    assert result.clarification_needed is True


def test_openai_client_initialization_failure_returns_clarification(monkeypatch):
    monkeypatch.setattr("llm.request_parser.load_dotenv", lambda: None)
    monkeypatch.setattr(
        "llm.request_parser.OpenAI",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("invalid client configuration")),
    )
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-key")

    result = parse_request("Can I cycle in Bhopal today?")

    assert result.failure == RequestUnderstandingFailure.OPENAI_ERROR
    assert result.clarification_needed is True