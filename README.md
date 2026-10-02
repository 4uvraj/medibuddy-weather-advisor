# Weather-Advisory Support Bot

A weather-aware assistant that provides outdoor-activity guidance only from explicit, reviewed Standard Operating Procedures (SOPs).

## Architecture

```text
User
  -> OpenAI structured request extraction
  -> LangGraph orchestration
  -> Open-Meteo geocoding and forecast
  -> deterministic policy engine
  -> SOP match / no-match / conflict
  -> grounded response
```

- **OpenAI is used only to understand a request** and extract activity, location, and requested time period into a constrained schema. Extracted activities are validated against the application's controlled taxonomy.
- **OpenAI does not decide safety**, select SOPs, create directives, or provide weather values.
- **Weather facts come from Open-Meteo.** Provider responses are checked and normalized before policy evaluation or display. Open-Meteo needs no API key.
- **Safety guidance comes from external YAML SOPs** in `policies/sops.yaml`. The deterministic policy engine decides applicability and returns traceable matches.
- **No applicable SOP means no invented advice.** The answer states that no applicable SOP guidance was found.
- LangGraph has real conditional branches for clarification, geocoding outcomes, weather outcomes, no-match, matches, and policy conflicts.
- API, location, malformed-weather, and unsupported-period failures do not produce weather claims or safety conclusions.

## SOPs

SOPs are versioned YAML records validated when the graph loads the policy set. Conditions use an allowlisted set of weather fields, canonical units, and deterministic operators. Every applicable SOP is retained and sorted by severity, priority, then stable policy ID. Explicit unresolved conflicts produce a policy-conflict response instead of a guessed winner.

To add SOP #11 (or any later SOP), add a valid record to `policies/sops.yaml`, update the policy-set version in both YAML files, and restart the service. The generic loader and matcher discover it automatically; no graph or control-flow code changes are needed. For example, a new wind policy can declare `weather.wind_speed_10m`, `gte`, a canonical `km/h` threshold, and its approved directive in YAML.

## Failure and Safety Handling

- Missing or ambiguous request slots result in a clarification question. Follow-up slots are kept in process-local memory by session ID.
- A city with no geocoding result gets a not-found response; a geocoding service failure gets a location-unavailable response.
- Weather API errors or invalid/missing fields produce an honest forecast-unavailable response. No values are estimated or substituted.
- `NO_MATCH` explicitly says no applicable SOP guidance was found and adds no generic safety advice.
- A policy conflict is surfaced without choosing a recommendation.
- Prompt-injection text is treated as user data. The parser can only return the three request fields; all safety decisions still pass through the deterministic policy engine.

## Tests and Evaluations

The current regression suite has **96 pytest tests**. It includes policy schema/matcher tests, parser tests, mocked Open-Meteo tests, Streamlit session-context tests, and branching LangGraph tests. The evaluation runner checks exact and paraphrased SOP matches, no-match, forecast failure, and adversarial input using deterministic provider fixtures.

The severe-weather evaluation is the only live-weather case. On the last recorded run, Open-Meteo returned 23 Bhopal forecast samples but none matched a high/critical SOP; the case was honestly recorded as `NOT_APPLICABLE`, not forced to pass. The Streamlit smoke test submitted “Is it safe to cycle in Bhopal today?” through the UI and received live forecast facts plus the explicit no-SOP response. Evaluation request slots are injected as fixtures, so the deterministic evaluation runner does not score the live OpenAI extraction quality.

## Run Locally

From the repository root on Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file (it is ignored by Git) with your own OpenAI API key and model selection:

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
```

Never commit `.env` or paste a real key into source, tests, logs, or documentation. Open-Meteo is unauthenticated.

Run checks and start the chat app:

```powershell
python -m pytest -q
python -m evals.run_evals
streamlit run streamlit_app.py
```

The app creates an in-memory session ID per browser session. Conversation context is lost when the process restarts and is not shared between service instances.

## Deployment

`render.yaml` configures one Render web service. It installs `requirements.txt`, binds Streamlit to Render's assigned port, and declares `OPENAI_API_KEY` as a dashboard-provided secret (`sync: false`). Set `OPENAI_MODEL` in the service environment if a different supported model is desired. Open-Meteo requires no credential. The service uses in-memory session context, so deploy one instance for consistent follow-up behavior.
