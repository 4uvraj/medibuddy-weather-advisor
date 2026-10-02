from uuid import uuid4

import streamlit as st

from policy_engine.loader import load_policy_set
from ui.presentation import format_metric_value, parse_sop_advisories, parse_weather_display
from workflow.graph import build_weather_graph


EXAMPLE_PROMPTS = (
    "Is it safe to cycle in Bhopal today?",
    "Is running safe in Bhopal today?",
    "Can I go cycling this evening?",
    "Can I cycle in Indore tomorrow?",
)


def _render_response(message: dict[str, str]) -> None:
    response = message["content"]
    status = message.get("status", "")
    weather = parse_weather_display(response)

    if status in {"weather_error", "location_error", "location_not_found"}:
        st.warning(response, icon="⚠️")
        return

    if weather is not None:
        st.caption(f"Weather information · {weather.location} · {weather.period} · {weather.timezone}")
        primary_metrics = st.columns(2)
        primary_metrics[0].markdown(
            f"**Temperature**  \n{format_metric_value(weather.temperature, 'temperature')}"
        )
        primary_metrics[1].markdown(f"**Wind**  \n{format_metric_value(weather.wind, 'wind')}")
        secondary_metrics = st.columns(3)
        secondary_metrics[0].markdown(f"**Rain**  \n{weather.precipitation}")
        secondary_metrics[1].markdown(f"**Rain chance**  \n{weather.precipitation_probability}")
        secondary_metrics[2].markdown(f"**UV index**  \n{weather.uv}")
        st.caption(f"Conditions: {weather.conditions}")

        if status == "no_sop":
            st.info("No applicable SOP", icon="ℹ️")
            st.write("No safety guidance was triggered by the written SOPs for these weather conditions.")
        elif status == "matched_sop":
            st.markdown("**SOP-based advisory**")
            policies = load_policy_set("policies/manifest.yaml")
            severity_by_id = {policy.policy_id: policy.severity.value for policy in policies.policies}
            for advisory in parse_sop_advisories(weather.advisory, severity_by_id):
                st.markdown(f"**[{advisory.severity}] {advisory.policy_id}**")
                st.write(advisory.directive)
        elif status == "policy_conflict":
            st.warning(weather.advisory)
        return

    if status == "clarification":
        st.info(response)
    else:
        st.write(response)


st.set_page_config(page_title="Weather Advisory Assistant", page_icon="🌦️")
st.info(
    "**Evaluator Notice**\n\n"
    "If you are reviewing this on **Render**, Open-Meteo's 10,000 req/day limit is frequently exhausted by Render's shared Free Tier IPs.\n\n"
    "If weather requests fail, please test the fully working distributed deployment at:\n\n"
    "👉 **[Streamlit Cloud Deployment](https://medibuddy-weather-advisor-evqet8axhe3vd9urfwday8.streamlit.app/)**",
    icon="🚨"
)
st.title("🌦️ Weather Advisory Assistant")
st.caption("Live weather + rule-based safety guidance")

with st.expander("How it works", expanded=False):
    st.markdown(
        "- Understand activity, location and time\n"
        "- Fetch live weather from Open-Meteo\n"
        "- Check written safety SOPs\n"
        "- Provide guidance only when an applicable SOP exists"
    )

st.markdown("**Try an example**")
example_columns = st.columns(2)
for index, example in enumerate(EXAMPLE_PROMPTS):
    if example_columns[index % 2].button(example, key=f"example_{index}", use_container_width=True):
        st.session_state.pending_prompt = example

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "weather_graph" not in st.session_state:
    st.session_state.weather_graph = build_weather_graph()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_response(message)
        else:
            st.markdown(message["content"])

submitted_prompt = st.chat_input("Ask about outdoor activity weather")
prompt = submitted_prompt or st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the forecast and applicable guidance..."):
            result = st.session_state.weather_graph.invoke(
                {"message": prompt, "session_id": st.session_state.session_id}
            )
        assistant_message = {
            "role": "assistant",
            "content": result["response"],
            "status": result.get("response_status", ""),
        }
        _render_response(assistant_message)
    st.session_state.messages.append(
        assistant_message
    )