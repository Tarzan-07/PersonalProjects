import streamlit as st
import json
import generator
import requests
st.title("Intent Classification")

input_text = st.text_input("Enter text for classification:")

st.button("Generate")
if st.button:
    response = requests.get('http://localhost:8000/joke', json=input_text)
    st.write(response.text)