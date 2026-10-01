# External Weather SOPs

Policies are externalized so reviewers can inspect, version, and change safety guidance without modifying policy-engine code, graph control flow, weather retrieval, or response composition. The matcher evaluates policy data only; it neither calls an LLM nor generates advice.

## Files and schema

`manifest.yaml` declares the policy-set version, update date, description, and policy document filename. `sops.yaml` contains the matching `policy_set_version` and a list of SOP records.

Every SOP has a stable, human-readable `policy_id`, semantic `version`, `category`, one or more canonical `activities`, `severity`, non-negative `priority`, effective dates, `conditions`, approved `directive`, `rationale`, and `source`. Optional `audiences` constrain a policy to a declared audience; optional `conflicts_with` declares an unresolved conflict with another policy ID. IDs are authored in YAML, never generated during evaluation.

Each condition has an allowlisted weather `field`, an `operator`, a scalar or list `value`, and (for numeric fields) its exact canonical `unit`. A policy can use an `all` group, an `any` group, or both. When both are present, every `all` condition and at least one `any` condition must match. Missing weather values never match.

## Taxonomy and aliases

Canonical activities are `cycling`, `running`, `walking`, `hiking`, `picnic`, `travel`, `outdoor_play`, and `water_activity`. Audiences are `children`, `older_adult`, and `chronic_condition`. Categories are `outdoor_exercise`, `travel`, `children`, `vulnerable_groups`, and `general_outdoor_activity`.

The small application alias map includes `bike ride`/`biking` → `cycling`, `jog`/`jogging` → `running`, `stroll` → `walking`, and `trekking` → `hiking`. Aliases normalize request input only; policy YAML must use canonical values. The application must validate any proposed activity against the taxonomy before constructing a policy request.

## Fields, units, and operators

| Policy field                        | Type        | Canonical unit / values                                             |
| ----------------------------------- | ----------- | ------------------------------------------------------------------- |
| `weather.temperature_2m`            | Numeric     | `degC`                                                              |
| `weather.wind_speed_10m`            | Numeric     | `km/h`                                                              |
| `weather.precipitation`             | Numeric     | `mm`                                                                |
| `weather.precipitation_probability` | Numeric     | `%`                                                                 |
| `weather.uv_index`                  | Numeric     | `index`                                                             |
| `weather.wmo_condition`             | Categorical | `clear`, `cloudy`, `fog`, `rain`, `snow`, `thunderstorm`, `unknown` |

The provider adapter must normalize numeric units before creating `NormalizedWeather`; the policy engine does no implicit conversion. The WMO weather code is mapped deterministically to the categorical signal (`thunderstorm` for WMO codes 95, 96, and 99, among other documented category mappings in the Python normalizer). Fuzzy or descriptive weather conditions therefore use a controlled categorical value, never LLM judgment.

Supported operators are `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, and `not_in`. Ordered comparisons are numeric-only. `in` and `not_in` require a non-empty list; other operators require a scalar. Unknown fields, operators, categories, activities, severities, invalid types, and mismatched units fail loading.

## Severity, ordering, and conflicts

Severity ranks are `low` < `moderate` < `high` < `critical`. Every applicable policy is returned; none is dropped because another has higher severity. Results sort by severity descending, priority descending, then stable `policy_id` ascending. Priority is a non-negative integer, and larger values sort first within the same severity.

If two matching policies explicitly list each other in `conflicts_with`, evaluation returns `POLICY_CONFLICT` with both matches and the conflicting ID pair. Conflict declarations must be reciprocal and refer to policies in the same policy set. This version does not guess a winner; a reviewer must revise the policies or add a future explicit resolution design.

When no policy matches, the result is `NO_MATCH` with an empty match list. The engine does not provide generic safety advice; the later graph can render its approved no-guidance message.

## Add SOP #13

1. Add a complete SOP record to `sops.yaml` with a new stable ID, valid canonical category/activity, severity, dates, conditions, directive, rationale, and source.
2. Keep `manifest.yaml` and `sops.yaml` policy-set versions equal. Increment the policy-set version and update `updated_at` when publishing a reviewed change; increment the SOP version when that SOP changes.
3. Restart the application. Startup loading validates the manifest, document, and every SOP and fails clearly on invalid configuration.
4. The generic matcher loads the new record automatically. No Python, graph, weather, or response-generation code changes are required.

For example, an SOP using `weather.wind_speed_10m gte 45 km/h` becomes evaluable after adding its YAML record and restarting. A unit test demonstrates this data-only extension behavior.
