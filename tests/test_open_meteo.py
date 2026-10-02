from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from providers.open_meteo import (
    FORECAST_URL,
    GEOCODING_URL,
    ForecastSamplesUnavailable,
    GeocodingUnavailable,
    LocationNotFound,
    OpenMeteoProvider,
    ResolvedLocation,
    UnsupportedTimePeriod,
    WeatherUnavailable,
)
from policy_engine.weather import WeatherCondition


FIXED_NOW = datetime(2026, 10, 2, 8, 30, tzinfo=ZoneInfo("UTC"))


def forecast_payload():
    times = [f"2026-10-02T{hour:02d}:00" for hour in (7, 9, 10, 17, 18)]
    return {
        "timezone": "UTC",
        "hourly_units": {
            "temperature_2m": "°C",
            "wind_speed_10m": "km/h",
            "precipitation": "mm",
            "precipitation_probability": "%",
            "uv_index": "",
            "weather_code": "wmo code",
        },
        "hourly": {
            "time": times,
            "temperature_2m": [18, 20, 22, 25, 24],
            "wind_speed_10m": [20, 30, 45, 35, 25],
            "precipitation": [0, 0, 0, 1, 0],
            "precipitation_probability": [0, 10, 20, 60, 20],
            "uv_index": [0, 2, 5, 1, 0],
            "weather_code": [0, 1, 3, 61, 95],
        },
    }


def make_provider(handler):
    return OpenMeteoProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_geocode_and_weather_request_explicit_fields_and_normalize_units():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(
                200,
                json={"results": [{"name": "Bhopal", "latitude": 23.2599, "longitude": 77.4126}]},
            )
        return httpx.Response(200, json=forecast_payload())

    provider = make_provider(handler)
    location = provider.resolve_location("Bhopal")
    period = provider.fetch_weather(location, "today", now=FIXED_NOW)

    assert calls[0].url == httpx.URL(GEOCODING_URL).copy_with(
        query=b"name=Bhopal&count=1&language=en&format=json"
    )
    assert calls[1].url.host == httpx.URL(FORECAST_URL).host
    assert calls[1].url.path == httpx.URL(FORECAST_URL).path
    params = dict(calls[1].url.params)
    assert params["hourly"] == "temperature_2m,wind_speed_10m,precipitation,precipitation_probability,uv_index,weather_code"
    assert params["temperature_unit"] == "celsius"
    assert params["wind_speed_unit"] == "kmh"
    assert params["precipitation_unit"] == "mm"
    assert params["timezone"] == "auto"
    assert location == ResolvedLocation(name="Bhopal", latitude=23.2599, longitude=77.4126)
    assert [sample.time.hour for sample in period.samples] == [9, 10, 17, 18]
    assert period.samples[-1].weather.wmo_condition == WeatherCondition.THUNDERSTORM
    provider.close()


def test_city_not_found_is_distinct_from_provider_failure():
    provider = make_provider(lambda request: httpx.Response(200, json={"results": []}))

    with pytest.raises(LocationNotFound):
        provider.resolve_location("Unknown City")
    provider.close()


def test_geocoding_http_failure_is_reported_as_provider_failure():
    provider = make_provider(lambda request: httpx.Response(503))

    with pytest.raises(GeocodingUnavailable):
        provider.resolve_location("Bhopal")
    provider.close()


def test_weather_http_failure_is_reported_without_weather_values():
    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(
                200,
                json={"results": [{"name": "Bhopal", "latitude": 23.2, "longitude": 77.4}]},
            )
        return httpx.Response(503)

    provider = make_provider(handler)
    location = provider.resolve_location("Bhopal")

    with pytest.raises(WeatherUnavailable):
        provider.fetch_weather(location, "today", now=FIXED_NOW)
    provider.close()


def test_unrecognized_requested_period_is_not_guessed():
    provider = make_provider(lambda request: httpx.Response(200, json=forecast_payload()))
    location = ResolvedLocation(name="Bhopal", latitude=23.2, longitude=77.4)

    with pytest.raises(UnsupportedTimePeriod):
        provider.fetch_weather(location, "sometime soon", now=FIXED_NOW)
    provider.close()


def test_unexpected_provider_units_are_rejected():
    payload = forecast_payload()
    payload["hourly_units"]["wind_speed_10m"] = "m/s"
    provider = make_provider(lambda request: httpx.Response(200, json=payload))
    location = ResolvedLocation(name="Bhopal", latitude=23.2, longitude=77.4)

    with pytest.raises(WeatherUnavailable, match="unexpected unit"):
        provider.fetch_weather(location, "today", now=FIXED_NOW)
    provider.close()


def test_incomplete_weather_sample_is_rejected_instead_of_filled_in():
    payload = forecast_payload()
    payload["hourly"]["temperature_2m"][1] = None
    provider = make_provider(lambda request: httpx.Response(200, json=payload))
    location = ResolvedLocation(name="Bhopal", latitude=23.2, longitude=77.4)

    with pytest.raises(WeatherUnavailable, match="invalid hourly weather data"):
        provider.fetch_weather(location, "today", now=FIXED_NOW)
    provider.close()


def test_evening_with_available_forecast_samples_returns_weather():
    provider = make_provider(lambda request: httpx.Response(200, json=forecast_payload()))
    location = ResolvedLocation(name="Bhopal", latitude=23.2, longitude=77.4)
    now = datetime(2026, 10, 2, 16, 30, tzinfo=ZoneInfo("UTC"))

    period = provider.fetch_weather(location, "evening", now=now)

    assert [sample.time.hour for sample in period.samples] == [17, 18]
    assert period.label == "evening"
    provider.close()


def test_elapsed_evening_returns_availability_not_unsupported_period():
    provider = make_provider(lambda request: httpx.Response(200, json=forecast_payload()))
    location = ResolvedLocation(name="Bhopal", latitude=23.2, longitude=77.4)
    now = datetime(2026, 10, 2, 22, 30, tzinfo=ZoneInfo("UTC"))

    with pytest.raises(ForecastSamplesUnavailable) as error:
        provider.fetch_weather(location, "evening", now=now)

    assert error.value.user_message == (
        "No remaining evening forecast is available for today. Please try tomorrow evening."
    )
    provider.close()


def test_429_followed_by_success_retries_and_succeeds():
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"results": [{"name": "Bhopal", "latitude": 23.2, "longitude": 77.4}]})
    ]
    def handler(request):
        return responses.pop(0)

    provider = make_provider(handler)
    loc = provider.resolve_location("Bhopal")
    assert loc.name == "Bhopal"
    assert len(responses) == 0
    provider.close()


def test_persistent_429_eventually_returns_weather_unavailable():
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "0"})

    provider = make_provider(handler)
    with pytest.raises(GeocodingUnavailable):
        provider.resolve_location("Bhopal")
    provider.close()


def test_normal_non_429_4xx_is_not_repeatedly_retried():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(400)

    provider = make_provider(handler)
    with pytest.raises(GeocodingUnavailable):
        provider.resolve_location("Bhopal")

    assert len(calls) == 1  # No retries for 400
    provider.close()


def test_repeated_weather_queries_make_fresh_requests():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=forecast_payload())

    provider = make_provider(handler)
    loc = ResolvedLocation(name="Bhopal", latitude=23.2, longitude=77.4)

    provider.fetch_weather(loc, "today", now=FIXED_NOW)
    assert len(calls) == 1

    provider.fetch_weather(loc, "today", now=FIXED_NOW)
    assert len(calls) == 2  # No cache, fresh call made
    provider.close()
