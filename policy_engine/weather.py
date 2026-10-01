from enum import StrEnum

from pydantic import BaseModel, ConfigDict, FiniteFloat


class WeatherCondition(StrEnum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    FOG = "fog"
    RAIN = "rain"
    SNOW = "snow"
    THUNDERSTORM = "thunderstorm"
    UNKNOWN = "unknown"


class NormalizedWeather(BaseModel):
    """Weather fields after provider-unit and categorical normalization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature_2m: FiniteFloat | None = None
    wind_speed_10m: FiniteFloat | None = None
    precipitation: FiniteFloat | None = None
    precipitation_probability: FiniteFloat | None = None
    uv_index: FiniteFloat | None = None
    wmo_condition: WeatherCondition | None = None


def normalize_wmo_code(code: int) -> WeatherCondition:
    """Map Open-Meteo/WMO weather codes to stable categories without an LLM."""
    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("WMO weather code must be an integer")
    if code == 0:
        return WeatherCondition.CLEAR
    if code in {1, 2, 3}:
        return WeatherCondition.CLOUDY
    if code in {45, 48}:
        return WeatherCondition.FOG
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return WeatherCondition.RAIN
    if code in {71, 73, 75, 77, 85, 86}:
        return WeatherCondition.SNOW
    if code in {95, 96, 99}:
        return WeatherCondition.THUNDERSTORM
    return WeatherCondition.UNKNOWN