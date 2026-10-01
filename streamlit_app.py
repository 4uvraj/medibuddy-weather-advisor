from uuid import uuid4

import streamlit as st

from workflow.graph import build_weather_graph


st.set_page_config(page_title="MediBuddy Weather Advisor", page_icon="☁️")
st.title("MediBuddy Weather Advisor")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "weather_graph" not in st.session_state:
    st.session_state.weather_graph = build_weather_graph()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about outdoor activity weather"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the forecast and applicable guidance..."):
            result = st.session_state.weather_graph.invoke(
                {"message": prompt, "session_id": st.session_state.session_id}
            )
        st.markdown(result["response"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["response"]}
    )