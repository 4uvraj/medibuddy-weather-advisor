from ui.presentation import format_metric_value, parse_sop_advisories, parse_weather_display


WEATHER_RESPONSE = (
    "Open-Meteo forecast for Bhopal, today (Asia/Kolkata): temperature 22.1 to 32.1 C; "
    "wind 0.3 to 10.7 km/h; total precipitation 0 mm; precipitation probability up to 0%; "
    "UV index up to 7.5; conditions clear (WMO code 0)."
)


def test_weather_response_is_split_into_display_fields_without_rewriting_values():
    parsed = parse_weather_display(WEATHER_RESPONSE)

    assert parsed is not None
    assert parsed.location == "Bhopal"
    assert parsed.period == "today"
    assert parsed.timezone == "Asia/Kolkata"
    assert parsed.temperature == "22.1 to 32.1 C"
    assert parsed.wind == "0.3 to 10.7 km/h"
    assert parsed.precipitation == "0 mm"
    assert parsed.precipitation_probability == "0%"
    assert parsed.uv == "7.5"
    assert parsed.conditions == "clear (WMO code 0)"
    assert parsed.advisory == ""


def test_matched_response_keeps_sop_directives_as_advisory_text():
    response = (
        f"{WEATHER_RESPONSE}\nApplicable SOP guidance:\n"
        "- CYCLING-WIND v1.0.0: Postpone the ride."
    )

    parsed = parse_weather_display(response)

    assert parsed is not None
    assert parsed.advisory == "Applicable SOP guidance:\n- CYCLING-WIND v1.0.0: Postpone the ride."


def test_no_sop_response_is_parsed_as_weather_not_safety_advice():
    response = f"{WEATHER_RESPONSE}\nNo applicable SOP guidance was found for cycling in Bhopal during today."

    parsed = parse_weather_display(response)

    assert parsed is not None
    assert parsed.advisory == "No applicable SOP guidance was found for cycling in Bhopal during today."


def test_failure_or_clarification_text_is_not_treated_as_weather():
    assert parse_weather_display("I couldn't retrieve a verified forecast.") is None
    assert parse_weather_display("Please clarify the location.") is None


def test_metric_values_are_compact_and_preserve_ranges_and_units():
    assert format_metric_value("23.7 to 23.7 C", "temperature") == "23.7 °C"
    assert format_metric_value("24.5 to 35.1 C", "temperature") == "24.5–35.1 °C"
    assert format_metric_value("3.3 to 3.3 km/h", "wind") == "3.3 km/h"
    assert format_metric_value("7.6 to 13.7 km/h", "wind") == "7.6–13.7 km/h"


def test_sop_display_keeps_directive_and_adds_yaml_severity():
    advisory = (
        "Applicable SOP guidance:\n"
        "- OUTDOOR-EXERCISE-HEAT v1.0.0: "
        "Reschedule strenuous exercise to a cooler period or choose an indoor alternative."
    )

    parsed = parse_sop_advisories(advisory, {"OUTDOOR-EXERCISE-HEAT": "high"})

    assert len(parsed) == 1
    assert parsed[0].severity == "high"
    assert parsed[0].policy_id == "OUTDOOR-EXERCISE-HEAT"
    assert parsed[0].directive == (
        "Reschedule strenuous exercise to a cooler period or choose an indoor alternative."
    )