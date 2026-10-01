from enum import StrEnum


class Activity(StrEnum):
    CYCLING = "cycling"
    RUNNING = "running"
    WALKING = "walking"
    HIKING = "hiking"
    PICNIC = "picnic"
    TRAVEL = "travel"
    OUTDOOR_PLAY = "outdoor_play"
    WATER_ACTIVITY = "water_activity"


class Audience(StrEnum):
    CHILDREN = "children"
    OLDER_ADULT = "older_adult"
    CHRONIC_CONDITION = "chronic_condition"


class PolicyCategory(StrEnum):
    OUTDOOR_EXERCISE = "outdoor_exercise"
    TRAVEL = "travel"
    CHILDREN = "children"
    VULNERABLE_GROUPS = "vulnerable_groups"
    GENERAL_OUTDOOR_ACTIVITY = "general_outdoor_activity"


_ACTIVITY_ALIASES: dict[str, Activity] = {
    "bike ride": Activity.CYCLING,
    "bike riding": Activity.CYCLING,
    "biking": Activity.CYCLING,
    "jog": Activity.RUNNING,
    "jogging": Activity.RUNNING,
    "stroll": Activity.WALKING,
    "trekking": Activity.HIKING,
    "outdoor play": Activity.OUTDOOR_PLAY,
    "water sports": Activity.WATER_ACTIVITY,
}


def normalize_activity(value: str) -> Activity:
    """Normalize a small alias set and reject values outside the taxonomy."""
    normalized = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
    try:
        return Activity(normalized.replace(" ", "_"))
    except ValueError:
        try:
            return _ACTIVITY_ALIASES[normalized]
        except KeyError as error:
            allowed = ", ".join(activity.value for activity in Activity)
            raise ValueError(f"Unsupported activity {value!r}; expected one of: {allowed}") from error