import streamlit as st
import requests
import json

def response_generator(prompt, max_new_tokens=1024):
    url = "http://127.0.0.1:8000/query-stream/"
    data = {'query': prompt,
            "max_new_tokens": max_new_tokens}

    with requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"}, stream=True) as r:
        for chunk in r.iter_content(10, decode_unicode=True):
            yield chunk


st.title("Test Streaming App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(""):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
