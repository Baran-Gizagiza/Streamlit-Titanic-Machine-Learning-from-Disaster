import pandas as pd
import streamlit as st

class Sidebar_value():
    def bins_value(self):
        bins = st.sidebar.number_input(
            '(3)-2. Please select "bins" for Fare',
            min_value=2,
            max_value=100,
            value=10
        )
        return bins

    def solution_value(self):
        solution = st.sidebar.selectbox(
            '(3)-3. Please select your solution for "Age"',
            ["Exclude", "Mean+STD", "Mean", "Median"]
        )
        return solution

    def test_size_value(self):
        test_size = st.sidebar.number_input(
        '(4) Please select "test_size"',
        min_value=0.1,
        max_value=0.9,
        value=0.2
        )
        return test_size

    def file_output_value(self):
        file_output = st.sidebar.radio(
            'Do you want to output cvs file?"',
            ['No', 'Yes']
            )
        return file_output


    def analysis_value(self):
        analysis = st.sidebar.button(
            'Analyze',
        )
        return analysis
