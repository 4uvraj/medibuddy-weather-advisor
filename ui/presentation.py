import re
from dataclasses import dataclass


_WEATHER_LINE = re.compile(
    r"^Open-Meteo forecast for (?P<location>.+), (?P<period>.+) "
    r"\((?P<timezone>[^()]*)\): temperature (?P<temperature>[^;]+); "
    r"wind (?P<wind>[^;]+); total precipitation (?P<precipitation>[^;]+); "
    r"precipitation probability up to (?P<precipitation_probability>[^;]+); "
    r"UV index up to (?P<uv>[^;]+); conditions (?P<conditions>.+)\.$"
)


@dataclass(frozen=True)
class WeatherDisplay:
    location: str
    period: str
    timezone: str
    temperature: str
    wind: str
    precipitation: str
    precipitation_probability: str
    uv: str
    conditions: str
    advisory: str


@dataclass(frozen=True)
class SopDisplay:
    severity: str
    policy_id: str
    directive: str


def parse_weather_display(response: str) -> WeatherDisplay | None:
    """Split the graph's existing weather line from its unchanged advisory text."""
    weather_line, separator, advisory = response.partition("\n")
    match = _WEATHER_LINE.fullmatch(weather_line)
    if match is None:
        return None
    return WeatherDisplay(
        **match.groupdict(),
        advisory=advisory if separator else "",
    )


def format_metric_value(value: str, kind: str) -> str:
    """Shorten redundant equal endpoints and normalize range/unit typography."""
    value = value.strip()
    patterns = {
        "temperature": (r"^(\d+(?:\.\d+)?) to \1 C$", r"^(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) C$", " °C"),
        "wind": (r"^(\d+(?:\.\d+)?) to \1 km/h$", r"^(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) km/h$", " km/h"),
    }
    if kind not in patterns:
        return value
    equal_pattern, range_pattern, unit = patterns[kind]
    equal_match = re.fullmatch(equal_pattern, value)
    if equal_match:
        return f"{equal_match.group(1)}{unit}"
    range_match = re.fullmatch(range_pattern, value)
    if range_match:
        return f"{range_match.group(1)}–{range_match.group(2)}{unit}"
    return value


def parse_sop_advisories(advisory: str, severity_by_id: dict[str, str]) -> tuple[SopDisplay, ...]:
    """Extract existing SOP IDs/directives for display, annotating their YAML severity."""
    matches = []
    for line in advisory.splitlines():
        match = re.fullmatch(r"- ([A-Z][A-Z0-9_-]*) v[0-9A-Za-z.+-]+: (.+)", line)
        if match is None:
            continue
        policy_id, directive = match.groups()
        matches.append(
            SopDisplay(
                severity=severity_by_id.get(policy_id, ""),
                policy_id=policy_id,
                directive=directive,
            )
        )
    return tuple(matches)