import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.app import recommendation
from src.visualization_app import visualization


def main():
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Recommendation System", "Visualization"])

    if page == "Homepage":
        st.title("Homepage")
        st.markdown(
            """
                This is a demo of a Streamlit app that shows Airbnb Recommendation System.
            """)
        st.header('A little bit background')


    elif page == "Recommendation System":
        recommendation()

    elif page == "Visualization":
        visualization()


if __name__ == "__main__":
    main()
