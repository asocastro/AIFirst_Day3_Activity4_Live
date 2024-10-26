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
   st.write("Welcome to the News Article Summarizer Tool, designed to provide you with clear, concise, and well-structured summaries of news articles. This tool is ideal for readers who want to quickly grasp the essential points of any news story without wading through lengthy articles. Whether you’re catching up on global events, diving into business updates, or following the latest political developments, this summarizer delivers all the important details in a brief, easily digestible format.")
   st.write("## What the Tool Does")
   st.write("The News Article Summarizer Tool reads and analyzes full-length news articles, extracting the most critical information and presenting it in a structured manner. It condenses lengthy pieces into concise summaries while maintaining the integrity of the original content. This enables users to quickly understand the essence of any news story.")
   st.write("## How It Works")
   st.write("The tool follows a comprehensive step-by-step process to create accurate and objective summaries:")
   st.write("*Analyze and Extract Information:* The tool carefully scans the article, identifying key elements such as the main event or issue, people involved, dates, locations, and any supporting evidence like quotes or statistics.")
   st.write("*Structure the Summary:* It organizes the extracted information into a clear, consistent format. This includes:")
   st.write("- *Headline:* A brief, engaging headline that captures the essence of the story.")
   st.write("- *Lead:* A short introduction summarizing the main event.")
   st.write("- *Significance:* An explanation of why the news matters.")
   st.write("- *Details:* A concise breakdown of the key points.")
   st.write("- *Conclusion:* A wrap-up sentence outlining future implications or developments.")
   st.write("# Why Use This Tool?")
   st.write("- *Time-Saving:* Quickly grasp the key points of any article without having to read through long pieces.")
   st.write("- *Objective and Neutral:* The tool maintains an unbiased perspective, presenting only factual information.")
   st.write("- *Structured and Consistent:* With its organized format, users can easily find the most relevant information, ensuring a comprehensive understanding of the topic at hand.")
   st.write("# Ideal Users")
   st.write("This tool is perfect for:")
   st.write("- Busy professionals who need to stay informed but have limited time.")
   st.write("- Students and researchers looking for quick, accurate summaries of current events.")
   st.write("- Media outlets that want to provide readers with quick takes on trending news.")
   st.write("Start using the News Article Summarizer Tool today to get concise and accurate insights into the news that matters most!")
   
elif options == "About Us":
     st.title('News Summarizer Tool')
     st.subheader("About Me")
     st.write("# Alexander Castro")
     st.image('images/pic.jpg', width=200)
     st.write("## AI First Bootcamp Student")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/alexander-sebastian-castro/")
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
                 response = requests.get(News_Article)
                 soup = BeautifulSoup(response.content, 'html.parser')
                 
                 # Extract text from paragraphs
                 paragraphs = soup.find_all('p')
                 article_text = ' '.join([p.get_text() for p in paragraphs])

                 # OpenAI-based summarization
                 System_Prompt = """You are a skilled news summarizer tasked with creating concise, informative summaries of news articles for a general audience. Use the RICCE framework to ensure each summary is clear, accurate, and engaging. Follow these guidelines:

Role (R): You are an expert in news summarization, presenting complex stories in a way that is quick to read and easy to understand for a wide audience.

Instructions (I): Summarize the main points of the article in 100 words or fewer, focusing on the key details: who, what, when, where, why, and any significant impacts. Ensure all relevant context is included so readers get a comprehensive understanding without needing additional background.

Context (C): The audience includes readers who want an efficient, reliable overview of current events. Write in simple language, avoiding jargon and technical terms, and maintain a neutral, fact-based tone throughout.

Constraints (C): Keep summaries short and relevant to the core message of the story. Avoid opinions, subjective language, and sensationalism. Structure each summary to start with the most crucial points and follow with any necessary supporting details.

Examples (E):

Example: “Hurricane Fiona hit Puerto Rico on Sunday, causing major flooding, power outages, and infrastructure damage. Officials report thousands of residents displaced as rescue efforts continue. The hurricane, now a Category 3 storm, is expected to impact the Dominican Republic next, prompting widespread emergency preparations.”"""
                 user_message = f"Please summarize the following news article: {article_text}"
                 struct = [{'role': 'system', 'content': System_Prompt}]
                 struct.append({"role": "user", "content": user_message})
                 chat = openai.ChatCompletion.create(model="gpt-4-mini", messages=struct)
                 summary = chat.choices[0].message.content
                 struct.append({"role": "assistant", "content": summary})

                 st.success("Summary generated successfully!")
                 
                 st.subheader("Article Summary:")
                 st.write(summary)
             except Exception as e:
                 st.error(f"An error occurred: {str(e)}")
