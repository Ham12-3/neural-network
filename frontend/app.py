import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

SUMMARY_LENGTHS = {
    "Short (~30 words)": 30,
    "Medium (~80 words)": 80,
    "Long (~150 words)": 150,
}

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_texts"


def load_samples() -> dict[str, str]:
    samples = {}
    if SAMPLE_DIR.is_dir():
        for f in sorted(SAMPLE_DIR.glob("*.txt")):
            samples[f.stem.replace("_", " ").title()] = f.read_text(encoding="utf-8")
    return samples


def main():
    st.set_page_config(page_title="Text Summariser", page_icon="📝", layout="centered")
    st.title("Text Summariser")
    st.caption("Powered by a neural network summarisation model")

    samples = load_samples()

    # --- Sample selector ---
    if samples:
        choice = st.selectbox("Load a sample text", ["(none)"] + list(samples.keys()))
        if choice != "(none)":
            st.session_state["input_text"] = samples[choice]

    # --- Input area ---
    text = st.text_area(
        "Enter text to summarise",
        value=st.session_state.get("input_text", ""),
        height=250,
        key="input_text",
    )

    # --- Controls ---
    col1, col2 = st.columns([2, 1])
    with col1:
        length_label = st.selectbox("Summary length", list(SUMMARY_LENGTHS.keys()), index=1)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Summarise", type="primary", use_container_width=True)

    # --- Call API ---
    if run:
        if not text or not text.strip():
            st.warning("Please enter some text first.")
            return

        max_words = SUMMARY_LENGTHS[length_label]

        with st.spinner("Generating summary..."):
            try:
                resp = requests.post(
                    f"{API_URL}/summarise",
                    json={"text": text, "max_words": max_words},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.ConnectionError:
                st.error("Cannot reach the backend API. Is it running?")
                return
            except requests.HTTPError as exc:
                st.error(f"API error: {exc.response.text}")
                return

        st.subheader("Summary")
        st.write(data["summary"])

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Model", data["model"].split("/")[-1])
        c2.metric("Time", f"{data['took_ms']:.0f} ms")


if __name__ == "__main__":
    main()
