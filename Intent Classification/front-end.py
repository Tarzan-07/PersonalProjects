import streamlit as st
import json
from generator import JokeGenerator
import requests
st.title("Intent Classification")

input_text = st.text_input("Enter text for classification:")

st.button("Generate")
if st.button:
    # response = requests.get('http://localhost:8000/joke', json=input_text)
    response = JokeGenerator.generate_response(input_text)
    st.write(response.text)