from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from policy_engine.weather import WeatherCondition


class FieldKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class WeatherFieldSpec:
    attribute: str
    kind: FieldKind
    canonical_unit: str | None
    allowed_values: frozenset[str] = frozenset()


WEATHER_FIELD_SPECS = MappingProxyType(
    {
        "weather.temperature_2m": WeatherFieldSpec("temperature_2m", FieldKind.NUMERIC, "degC"),
        "weather.wind_speed_10m": WeatherFieldSpec("wind_speed_10m", FieldKind.NUMERIC, "km/h"),
        "weather.precipitation": WeatherFieldSpec("precipitation", FieldKind.NUMERIC, "mm"),
        "weather.precipitation_probability": WeatherFieldSpec(
            "precipitation_probability", FieldKind.NUMERIC, "%"
        ),
        "weather.uv_index": WeatherFieldSpec("uv_index", FieldKind.NUMERIC, "index"),
        "weather.wmo_condition": WeatherFieldSpec(
            "wmo_condition",
            FieldKind.CATEGORICAL,
            None,
            frozenset(condition.value for condition in WeatherCondition),
        ),
    }
)