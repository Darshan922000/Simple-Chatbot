import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", 
    json = {'input':{'topic':input_text}})

    return response.json()["output"]["content"]

def get_ollama_response(input_text1):
    response = requests.post("http://localhost:8000/poem/invoke", 
    json = {'input':{'topic':input_text1}})

    return response.json()["output"]

st.title("D9AI")
input_text = st.text_input("Essay Topic: ")
input_text1 = st.text_input("Poem Topic: ")

if input_text:
    st.write(get_openai_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))