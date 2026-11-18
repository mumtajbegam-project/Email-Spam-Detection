import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

st.title("📧 Email Spam Detection")
st.write("Check whether your message is Spam or Legitimate.")

message = st.text_area("Enter your message:")

if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        result = model.predict([message])[0]

        if result == "spam":
            st.error("⚠️ This is a SPAM message!")
        else:
            st.success("✅ This is a LEGITIMATE message!")