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
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

def scrape_website(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        # Send a GET request to the website with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from all <p> tags
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        
        return {"paragraphs": paragraphs}
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
    st.image('images/White_AI Republic.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Me", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :
    st.title('News Summarizer Tool')
    st.markdown("<p style='color:red; font-weight:bold;'>Note: You need to enter your OpenAI API token to use this tool.</p>", unsafe_allow_html=True)
    st.write("Welcome to the News Summarizer Tool, designed to give you quick, concise summaries of news articles. Ideal for busy readers, this tool distills the core points of news stories across politics, business, global events, and more.")
    st.write("## Features")
    st.write("Our summarizer identifies and extracts key information, presenting it in a structured format including a headline, summary, and key details. This provides a snapshot of any article in seconds.")
    st.write("## How It Works")
    st.write("1. **Scan & Analyze**: Key elements such as events, people, and data points are identified.")
    st.write("2. **Summarize**: Information is condensed into clear sections—headline, main points, significance, and future outlook.")
    st.write("## Benefits")
    st.write("- **Saves Time**: Grasp the essence of articles without wading through full texts.")
    st.write("- **Clear & Objective**: Unbiased, structured summaries let you get the facts fast.")
    st.write("## Ideal Users")
    st.write("Perfect for busy professionals, students, researchers, and anyone who wants fast, reliable news summaries.")

   
elif options == "About Me":
    st.title('News Summarizer Tool')
    st.subheader("About Me")
    st.write("# Alexander Castro")
    # col1, col2, col3 = st.columns([3, 2, 3])
    # with col1:
    #     st.image('images/pic.jpg', use_column_width=True)
    st.write("## AI First Bootcamp Student")
    st.text("Connect with me via Linkedin : https://www.linkedin.com/in/alexander-sebastian-castro/")
    st.text("Visit my Github: https://github.com/asocastro/")
    st.write("\n")


elif options == "Model" :
    st.title('News Summarizer Tool') 
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        News_Article = st.text_input("News Article URL", placeholder="Enter article URL: ")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary"):
             try:
                 # Fetch the article content
                 data = scrape_website(News_Article)

                 # OpenAI-based summarization
                 System_Prompt = """You are a skilled news summarizer tasked with creating concise, informative summaries of news articles for a general audience. Use the RICCE framework to ensure each summary is clear, accurate, and engaging. Follow these guidelines:

Role (R): You are an expert in news summarization, presenting complex stories in a way that is quick to read and easy to understand for a wide audience.

Instructions (I): Summarize the main points of the article in 150 words or fewer, focusing on the key details: who, what, when, where, why, and any significant impacts. Ensure all relevant context is included so readers get a comprehensive understanding without needing additional background. If the outseems nonsensical like only containing the words ADVERTISEMENT, recommend they use another news site.

Context (C): The audience includes readers who want an efficient, reliable overview of current events. Write in simple language, avoiding jargon and technical terms, and maintain a neutral, fact-based tone throughout.

Constraints (C): Keep summaries short and relevant to the core message of the story. Avoid opinions, subjective language, and sensationalism. Structure each summary to start with the most crucial points and follow with any necessary supporting details.

Examples (E):

Example: “Hurricane Fiona hit Puerto Rico on Sunday, causing major flooding, power outages, and infrastructure damage. Officials report thousands of residents displaced as rescue efforts continue. The hurricane, now a Category 3 storm, is expected to impact the Dominican Republic next, prompting widespread emergency preparations.

”"""
                 user_message = f"Please summarize the following news article: {data}"
                 struct = [{'role': 'system', 'content': System_Prompt}]
                 struct.append({"role": "user", "content": user_message})
                 chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
                 summary = chat.choices[0].message.content
                 struct.append({"role": "assistant", "content": summary})
                 
                 st.success("Summary generated successfully!")

                 st.subheader("Article Summary:")
                 st.write(summary)
             except Exception as e:
                 st.error(f"An error occurred: {str(e)}")
