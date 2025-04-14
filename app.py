import streamlit as st
import time
from app.model_manager import ModelManager

st.title("NU OGS Chatbot")

# Cache the model manager so that the model is loaded only once.
@st.cache_resource
def get_model_manager():
    return ModelManager()
with st.spinner("Loading OGS bot... please wait"):
    mm = get_model_manager()

# Streamed response emulator
def response_generator(query):
    response = mm.query_handler(query)

    def type_response(text):
        for word in text.split():
            yield word + " "
            time.sleep(0.02)

    if isinstance(response, str):
        yield from type_response(response)
    else:
        yield from type_response(response['output'])
        # After output, yield the URLs with HTML line breaks
        yield "<br><br>See the following links for more:<br>" + "<br>".join(response['urls'])


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask anything OGS-Related"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)

    # Instead of st.write_stream, use a placeholder and update with markdown
    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated_text = ""
        for token in response_generator(prompt):
            accumulated_text += token
            # Update the placeholder with markdown so that HTML (e.g. <br>) is rendered
            placeholder.markdown(accumulated_text, unsafe_allow_html=True)
            # time.sleep(0.02) is already used in type_response
        response = accumulated_text

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})