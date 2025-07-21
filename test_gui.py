import streamlit as st
import requests

st.set_page_config(page_title="CacheBot", page_icon="💬")

st.title(" CacheBot (with Redis Cache)")
st.markdown("Talk to the bot. Cached responses will be marked!")

session_id = st.text_input("Session ID", value="user123")
message = st.text_input("Your Message", "")

if st.button("Send"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        res = requests.post("http://localhost:8000/process_prompt", json={
            "session_id": session_id,
            "message": message
        })

        if res.status_code == 200:
            data = res.json()
            st.markdown(f"**Bot:** {data['response']}")
            if data['from_cache']:
                st.success(" Response from Redis Cache")
            else:
                st.info(" Freshly generated")
        else:
            st.error("Error: Could not reach backend.")
