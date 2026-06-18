"""
SmartQuery - AI Document QA Chatbot
Main application entry point
"""

import streamlit as st


def main():
    st.set_page_config(
        page_title="SmartQuery - AI Document QA",
        page_icon="🤖",
        layout="wide",
    )
    st.title("SmartQuery 🤖")
    st.write("AI-powered Document Q&A Chatbot")


if __name__ == "__main__":
    main()
