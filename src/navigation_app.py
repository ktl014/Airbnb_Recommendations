# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:50:37 2020

@author: dguan
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.app import recommendation
from src.visualization_app import visualization


def main():
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Data Analytics",
                                                  "Recommendation System"])

    if page == "Homepage":
        st.title("Airbnb Country Analytics & Recommendation")
        st.markdown(
            """
                The **purpose** of the **Airbnb Country Analytics & Recommendation 
                project** is to share knowledge on travelling patterns and make 
                accurate predictions for where a user may go for their first 
                experience. The larger implication of this work would be to decrease 
                average time to first booking and better forecast demand.\n\n
                This application provides
                - A **comprehensive analysis** of Airbnb's travelling data
                - A **vision** on how awesome machine learning can be for sharing 
                personalized content.
            """)
        st.header('The Magic of our Data Analytics & Recommendation System')
        st.markdown("The only way to truly understand how magical our application is "
                    "to play around with it. But if you need ot be convinced first, "
                    "please stay tuned for our presentation!")

    elif page == "Data Analytics":
        visualization()

    elif page == "Recommendation System":
        recommendation()


if __name__ == "__main__":
    main()
