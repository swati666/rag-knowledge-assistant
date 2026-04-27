import streamlit as st
import requests

API_URL = "https://rag-knowledge-assistant-sw3i.onrender.com/ask"

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖"
)

st.title("🤖 RAG Knowledge Assistant")


# -----------------------------
# Initialize chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

        if "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"{i}. {source}")


# -----------------------------
# User input
# -----------------------------
prompt = st.chat_input("Ask a question...")

if prompt:

    # Save user message
    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)


    # Call RAG API
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"question": prompt}
                )

                data = response.json()

                answer = data["answer"]
                sources = data["sources"]

                st.write(answer)

                with st.expander("Sources"):
                    for i, s in enumerate(sources,1):
                        st.write(f"{i}. {s}")

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role":"assistant",
                        "content":answer,
                        "sources":sources
                    }
                )

            except Exception as e:
                st.error(f"Error: {e}")