from datetime import date, datetime, timedelta
from math import isfinite
import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr

from policy_engine.weather import NormalizedWeather, normalize_wmo_code


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = (
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weather_code",
)
_EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "wind_speed_10m": "km/h",
    "precipitation": "mm",
    "precipitation_probability": "%",
    "uv_index": "",
}


class OpenMeteoError(RuntimeError):
    """Base error for provider failures safe to report to the caller."""


class LocationNotFound(OpenMeteoError):
    """Open-Meteo geocoding returned no matching location."""


class GeocodingUnavailable(OpenMeteoError):
    """Open-Meteo geocoding could not be completed or validated."""


class WeatherUnavailable(OpenMeteoError):
    """Open-Meteo weather data could not be fetched or validated."""


class UnsupportedTimePeriod(OpenMeteoError):
    """The requested time period is not supported by deterministic selection."""


class ForecastSamplesUnavailable(OpenMeteoError):
    """A valid forecast period has no currently available forecast samples."""

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__(user_message)


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResolvedLocation(ProviderModel):
    name: StrictStr = Field(min_length=1)
    latitude: FiniteFloat
    longitude: FiniteFloat


class ForecastSample(ProviderModel):
    time: datetime
    weather: NormalizedWeather
    source_weather_code: StrictInt


class WeatherPeriod(ProviderModel):
    label: StrictStr
    timezone: StrictStr
    samples: tuple[ForecastSample, ...] = Field(min_length=1)


_GLOBAL_CLIENT = httpx.Client(
    timeout=10.0,
    headers={"User-Agent": "WeatherAdvisoryBot/1.0 (https://github.com/4uvraj/medibuddy-weather-advisor)"}
)


class OpenMeteoProvider:
    """Small sync adapter for Open-Meteo geocoding and hourly forecast data."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client if client is not None else _GLOBAL_CLIENT
        self._owns_client = client is not None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def resolve_location(self, city: str) -> ResolvedLocation:

        payload = self._get_json(
            GEOCODING_URL,
            {"name": city, "count": 1, "language": "en", "format": "json"},
            error_type=GeocodingUnavailable,
        )
        results = payload.get("results")
        if results is None or results == []:
            raise LocationNotFound(f"No location found for {city!r}")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise GeocodingUnavailable("Open-Meteo returned an invalid geocoding response")

        result = results[0]
        try:
            return ResolvedLocation(
                name=result["name"],
                latitude=self._finite_number(result["latitude"], "latitude"),
                longitude=self._finite_number(result["longitude"], "longitude"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GeocodingUnavailable("Open-Meteo returned incomplete location data") from error

    def fetch_weather(
        self,
        location: ResolvedLocation,
        requested_time_period: str,
        *,
        now: datetime | None = None,
    ) -> WeatherPeriod:

        payload = self._get_json(
            FORECAST_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "hourly": ",".join(HOURLY_FIELDS),
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
                "timezone": "auto",
                "forecast_days": 7,
            },
            error_type=WeatherUnavailable,
        )
        return self._normalize_forecast(payload, requested_time_period, now=now)

    def _get_json(
        self,
        url: str,
        params: dict[str, object],
        *,
        error_type: type[OpenMeteoError],
    ) -> dict[str, Any]:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise error_type("Open-Meteo returned an invalid response")
                return payload
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 429 and attempt < max_retries:
                    retry_after = error.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = 1.0 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise error_type("Open-Meteo request failed") from error
            except httpx.RequestError as error:
                if attempt < max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise error_type("Open-Meteo request failed") from error
            except ValueError as error:
                raise error_type("Open-Meteo request failed") from error
        raise error_type("Open-Meteo request failed")

    @staticmethod
    def _finite_number(value: object, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Open-Meteo field {field!r} was not numeric")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"Open-Meteo field {field!r} was not finite")
        return number

    def _normalize_forecast(
        self,
        payload: dict[str, Any],
        requested_time_period: str,
        *,
        now: datetime | None,
    ) -> WeatherPeriod:
        timezone_name = payload.get("timezone")
        hourly = payload.get("hourly")
        units = payload.get("hourly_units")
        if not isinstance(timezone_name, str) or not isinstance(hourly, dict) or not isinstance(units, dict):
            raise WeatherUnavailable("Open-Meteo returned incomplete forecast metadata")
        for field, expected_unit in _EXPECTED_UNITS.items():
            if units.get(field) != expected_unit:
                raise WeatherUnavailable(f"Open-Meteo returned an unexpected unit for {field}")

        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise WeatherUnavailable("Open-Meteo returned an unknown timezone") from error

        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise WeatherUnavailable("Open-Meteo returned no hourly forecast times")
        arrays: dict[str, list[object]] = {}
        for field in HOURLY_FIELDS:
            values = hourly.get(field)
            if not isinstance(values, list) or len(values) != len(times):
                raise WeatherUnavailable(f"Open-Meteo returned incomplete hourly data for {field}")
            arrays[field] = values

        local_now = self._local_now(timezone, now)
        target_date, allowed_hours = self._select_window(requested_time_period, local_now)
        samples: list[ForecastSample] = []
        for index, raw_time in enumerate(times):
            if not isinstance(raw_time, str):
                raise WeatherUnavailable("Open-Meteo returned an invalid forecast timestamp")
            try:
                parsed_time = datetime.fromisoformat(raw_time)
            except ValueError as error:
                raise WeatherUnavailable("Open-Meteo returned an invalid forecast timestamp") from error
            local_time = (
                parsed_time.replace(tzinfo=timezone)
                if parsed_time.tzinfo is None
                else parsed_time.astimezone(timezone)
            )
            if local_time.date() != target_date or (
                allowed_hours is not None and local_time.hour not in allowed_hours
            ):
                continue
            if target_date == local_now.date() and local_time < local_now.replace(
                minute=0, second=0, microsecond=0
            ):
                continue

            try:
                source_code = arrays["weather_code"][index]
                if isinstance(source_code, bool) or not isinstance(source_code, int):
                    raise TypeError("weather_code must be an integer")
                weather = NormalizedWeather(
                    temperature_2m=self._finite_number(arrays["temperature_2m"][index], "temperature_2m"),
                    wind_speed_10m=self._finite_number(arrays["wind_speed_10m"][index], "wind_speed_10m"),
                    precipitation=self._finite_number(arrays["precipitation"][index], "precipitation"),
                    precipitation_probability=self._finite_number(
                        arrays["precipitation_probability"][index], "precipitation_probability"
                    ),
                    uv_index=self._finite_number(arrays["uv_index"][index], "uv_index"),
                    wmo_condition=normalize_wmo_code(source_code),
                )
                samples.append(
                    ForecastSample(
                        time=local_time,
                        weather=weather,
                        source_weather_code=source_code,
                    )
                )
            except (TypeError, ValueError) as error:
                raise WeatherUnavailable("Open-Meteo returned invalid hourly weather data") from error

        if not samples:
            day_part = self._day_part(requested_time_period)
            if target_date == local_now.date():
                if day_part:
                    message = (
                        f"No remaining {day_part} forecast is available for today. "
                        f"Please try tomorrow {day_part}."
                    )
                else:
                    message = "No remaining forecast is available for today. Please try tomorrow."
            else:
                message = f"No forecast samples are available for {requested_time_period}."
            raise ForecastSamplesUnavailable(message)
        return WeatherPeriod(
            label=requested_time_period,
            timezone=timezone_name,
            samples=tuple(samples),
        )

    @staticmethod
    def _local_now(timezone: ZoneInfo, now: datetime | None) -> datetime:
        if now is None:
            return datetime.now(timezone)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone)
        return now.astimezone(timezone)

    @staticmethod
    def _day_part(period: str) -> str | None:
        normalized = " ".join(period.casefold().split())
        for prefix in ("today ", "tomorrow ", "this "):
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix)
                break
        return normalized if normalized in {"morning", "afternoon", "evening", "night", "tonight"} else None

    @staticmethod
    def _select_window(period: str, now: datetime) -> tuple[date, frozenset[int] | None]:
        normalized = " ".join(period.casefold().split())
        day_part: str | None = None
        if normalized == "today" or normalized.startswith("today "):
            target_date = now.date()
            day_part = normalized.removeprefix("today").strip() or None
        elif normalized == "tomorrow" or normalized.startswith("tomorrow "):
            target_date = now.date() + timedelta(days=1)
            day_part = normalized.removeprefix("tomorrow").strip() or None
        elif normalized == "this" or normalized.startswith("this "):
            target_date = now.date()
            day_part = normalized.removeprefix("this").strip() or None
        elif normalized in {"morning", "afternoon", "evening", "night", "tonight"}:
            target_date = now.date()
            day_part = normalized
        else:
            try:
                target_date = date.fromisoformat(normalized)
            except ValueError as error:
                raise UnsupportedTimePeriod("The requested time period is not supported") from error

        hour_ranges = {
            "morning": frozenset(range(6, 12)),
            "afternoon": frozenset(range(12, 17)),
            "evening": frozenset(range(17, 22)),
            "night": frozenset(range(18, 24)),
            "tonight": frozenset(range(18, 24)),
        }
        if day_part is None:
            return target_date, None
        if day_part not in hour_ranges:
            raise UnsupportedTimePeriod("The requested time period is not supported")
        return target_date, hour_ranges[day_part]