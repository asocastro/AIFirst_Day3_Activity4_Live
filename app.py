import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

# Set page configuration
st.set_page_config(page_title="AI Assistant", layout="wide")

# Sidebar for OpenAI API key input
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    st.caption("Your API key is stored temporarily and not saved.")

# Main content
st.title("AI Assistant")

# About Me section
st.header("About Me")
st.write("""
Welcome to my AI Assistant app! I'm an AI enthusiast and developer passionate about creating 
user-friendly interfaces for AI interactions. This app demonstrates a simple integration 
with OpenAI's API to provide an interactive chat experience.
""")

# User input for the prompt
user_prompt = st.text_area("Enter your prompt here:", height=100)

# Button to generate response
if st.button("Generate Response"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not user_prompt:
        st.warning("Please enter a prompt.")
    else:
        try:
            # Set the OpenAI API key
            openai.api_key = openai_api_key
            
            # Generate response using OpenAI API
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=user_prompt,
                max_tokens=150
            )
            
            # Display the response
            st.subheader("AI Response:")
            st.write(response.choices[0].text.strip())
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.caption("Created with ❤️ using Streamlit and OpenAI")
