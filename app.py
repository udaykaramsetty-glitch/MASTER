import streamlit as st

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.write("Ask questions about company policies and HR documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_question = st.text_input("Enter your question:")

if st.button("Ask") and user_question:

    response = ask_bot(user_question)

    st.session_state.messages.append(
        {"question": user_question, "answer": response}
    )

for chat in st.session_state.messages:
    st.markdown(f"**Question:** {chat['question']}")
    st.markdown(f"**Answer:** {chat['answer']}")
    st.markdown("---")