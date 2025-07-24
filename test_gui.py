import streamlit as st
import requests
import json

# Configure the Streamlit page
st.set_page_config(page_title="CacheBot", page_icon="💬")
st.title(" CacheBot (with Redis + Semantic Matching)")
st.markdown("Cached responses show similarity")

# --- Backend URL Configuration ---
backend_url = st.text_input("Backend URL", value="http://localhost:8000")

# --- Fetch current similarity threshold from backend ---
default_threshold = 0.9
try:
    resp = requests.get(f"{backend_url}/cache/threshold")
    if resp.ok:
        default_threshold = resp.json().get("threshold", default_threshold)
except Exception:
    pass

# --- Similarity Threshold Controls (Sidebar) ---
threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.0, max_value=1.0,
    value=default_threshold, step=0.01
)
if st.sidebar.button("Set Threshold"):
    try:
        res = requests.post(f"{backend_url}/cache/threshold?value={threshold}")
        if res.ok:
            st.sidebar.success(f"Threshold set to {threshold}")
        else:
            st.sidebar.error(f"Error: {res.status_code} - {res.text}")
    except Exception as e:
        st.sidebar.error(f"Exception: {e}")

# --- Session ID Input ---
session_id = st.text_input("Session ID", "")

# --- Main Interaction: Send Prompt ---
prompt = st.text_input("Your Message (Prompt)", "")
if st.button(" Send Prompt"):
    if not backend_url or not session_id or not prompt.strip():
        st.warning("Please enter backend URL, session ID, and a prompt.")
    else:
        payload = {"session_id": session_id, "prompt": prompt, "message": prompt}
        try:
            res = requests.post(f"{backend_url}/process_prompt", json=payload)
            if res.ok:
                data = res.json()
                st.markdown(f"Bot: {data['response']}")
                if data.get("from_cache"):
                    sim = data.get("similarity")
                    if sim is not None:
                        st.success(f"From cache (similarity: {sim:.2f})")
                    else:
                        st.success(" From cache (exact match)")
                else:
                    st.info(" Freshly generated")
            else:
                st.error(f"Error: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"Exception: {e}")

# --- Cache Warming Section ---
st.markdown("---")
st.markdown("### Warm Cache")
warm_prompts_input = st.text_area(
    "Enter prompts to warm (one per line)",
    value=""
)
if st.button(" Warm Cache"):
    if not backend_url or not session_id:
        st.warning("Please enter backend URL and session ID before warming cache.")
    else:
        prompts_list = [line.strip() for line in warm_prompts_input.split("\n") if line.strip()]
        if not prompts_list:
            st.warning("Please enter at least one prompt to warm.")
        else:
            warm_payload = {
                "session_id": session_id,
                "prompts": prompts_list,
                "mode": "full"
            }
            try:
                wr = requests.post(f"{backend_url}/cache/warm", json=warm_payload)
                if wr.ok:
                    st.success(f"Cache warmed with {len(prompts_list)} prompts.")
                else:
                    st.error(f"Error warming cache: {wr.status_code} - {wr.text}")
            except Exception as e:
                st.error(f"Exception: {e}")
