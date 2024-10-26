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

warnings.filterwarnings("ignore")


st.set_page_config(page_title="Alex's News Summarizer Tool", page_icon="🏺", layout="wide")

st.markdown("""
<style>
    body {
        color: #5c4033;
        background-color: #f5e6d3;
    }
    .stApp {
        background-image: url('https://images.pexels.com/photos/1484776/pexels-photo-1484776.jpeg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stButton>button {
        color: #f5e6d3;
        background-color: #8b4513;
        border: 2px solid #5c4033;
    }
    .stTextInput>div>div>input {
        color: #5c4033;
        background-color: #f5e6d3;
    }
    .stTextArea>div>div>textarea {
        color: #5c4033;
        background-color: #f5e6d3;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar :
    st.image('')
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
        "🏺 Archaeological Dashboard", 
        ["Artifact Analysis", "Excavation Team", "Preservation Model", "AI Archaeologist", "Indiana Jones"],
        icons = ['search', 'geo-alt', 'tools', 'globe', 'compass'],
        menu_icon = "shovel", 
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
if options == "Artifact Analysis" :
    st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] > .main {
                background-image: url("https://images.pexels.com/photos/20872396/pexels-photo-20872396.jpeg");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style> """, 
        unsafe_allow_html=True)

    st.title('Amber\'s News Summarizer Tool')
    st.write("Welcome to the News Article Summarizer Tool, designed to provide you with clear, concise, and well-structured summaries of news articles. This tool is ideal for readers who want to quickly grasp the essential points of any news story without wading through lengthy articles. Whether you’re catching up on global events, diving into business updates, or following the latest political developments, this summarizer delivers all the important details in a brief, easily digestible format.")

   
elif options == "Excavation Team" :
     st.title('Excavation Team')
     st.subheader("About Us")
     st.write("# Amber Teng")
     st.image('AI_First_Day_3_Activity_4/images/jiu.gen_httpss.mj.run_12Txpfa5fM_a_portrait_of_a_24_year_old__5603fd8a-d5e6-46e5-9cf6-1267e0c864fd_0.png')
     st.write("## AI First Bootcamp Instructor")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/angelavteng/")
     st.write("\n")


elif options == "Preservation Model" :
     st.markdown("""
            <style>
            .stApp {
                background-image: url('https://images.pexels.com/photos/18934684/pexels-photo-18934684.jpeg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style> """, 
        unsafe_allow_html=True)
     st.title('Model')
     col1, col2, col3 = st.columns([1, 2, 1])

     with col2:
          News_Article = st.text_area("News Article", placeholder="News : ")
          submit_button = st.button("Generate Summary")

     if submit_button:
        with st.spinner("Generating Summary"):
             System_Prompt = """"You are a professional news article summarizer, trained to provide clear, concise, and informative summaries of news articles. Your objective is to extract and present the most crucial information in a structured format. Follow the steps below:

Step 1: Read and Analyze the Article Thoroughly
Read the entire article carefully to understand the overall context, main points, and any supporting information.
Pay attention to the 5Ws (Who, What, When, Where, Why) and the How. Focus on the main event or issue and identify key people, organizations, locations, dates, and any other relevant details.
Step 2: Extract Key Elements for the Summary
Main Event or Topic: Identify the core event, development, or issue that the article covers.
Context: Determine the background information or the circumstances surrounding the main event.
Key Figures: Highlight any important individuals, groups, or organizations involved.
Quotes and Evidence: Select one or two impactful quotes or pieces of evidence that strengthen the article's message.
Future Implications: Consider any mentioned consequences, future actions, or possible developments linked to the event.
Step 3: Structure the Summary
The summary should be concise but informative, following this structured format:

Headline: Craft a short, compelling headline (5-10 words) that captures the essence of the article.
Lead (1-2 sentences): Provide a brief introduction summarizing the main event or topic. Aim to cover the ‘What’ and ‘Who’ aspects here.
Why it Matters (1-2 sentences): Explain the significance or impact of the event. Why should the reader care about this news?
Details (2-3 sentences): Offer additional key points, such as evidence, quotes, or relevant background information that help explain the event further. Ensure this section includes important facts like ‘When’ and ‘Where.’
Zoom in (1-2 sentences): Dive into a specific element or perspective mentioned in the article that adds depth, such as a quote from an official or a unique angle on the issue.
Flashback (1 sentence): Provide a quick historical reference or a brief look back at related past events to give context.
Reality Check (1 sentence): Highlight any contrasting information or balance the report with another viewpoint if applicable.
Conclusion (1 sentence): Conclude with a sentence summarizing potential future actions, outcomes, or implications.
Step 4: Maintain Objectivity and Neutrality
Ensure that the summary is free of any bias or personal opinions. Present the information factually, with clarity and neutrality.
Use a professional and accessible tone, making the summary understandable even to readers unfamiliar with the topic.
Step 5: Format and Review the Summary
Double-check the summary to make sure it flows logically, is free of errors, and accurately reflects the key points of the article.
Verify that the length of each section is appropriate—keeping each segment brief and to the point while ensuring nothing critical is omitted.
Once you have processed the article following these steps, present the summary in the format outlined above."""
             user_message = News_Article
             struct = [{'role' : 'system', 'content' : System_Prompt}]
             struct.append({"role": "user", "content": user_message})
             chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
             response = chat.choices[0].message.content
             struct.append({"role": "assistant", "content": response})
             st.success("Insight generated successfully!")
             st.subheader("Summary : ")
             st.write(response)


elif options == "AI Archaeologist" :
     st.markdown("""
            <style>
            .stApp {
                background-image: url('https://images.pexels.com/photos/20872396/pexels-photo-20872396/free-photo-of-athens-at-night-with-a-view-of-the-parthenon-greece.png');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style> """, 
        unsafe_allow_html=True)
         
     st.title('AI Persona: Cultural Heritage Preservation and Oral History Curator Agent')
     col1, col2 = st.columns([1, 1])
    
     with col1:
        uploaded_file = st.file_uploader("Upload an image of cultural artifact", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
     with col2:
        text_input = st.text_area("Description of cultural artifact or practice", placeholder="Enter text here...")
    
     submit_button = st.button("Process Cultural Material")

     if submit_button:
        if uploaded_file is None and not text_input:
            st.warning("Please upload an image or provide a description (or both) before processing.")
        else:
            with st.spinner("Analyzing cultural material..."):
                System_Prompt = """
                You are a professional archaeologist trained to conduct detailed and thoughtful archaeological analyses. Your objective is to analyze artifacts, sites, and historical materials meticulously, revealing insights into past human societies, their cultures, and environments. Approach each analysis with thorough attention to cultural context, historical accuracy, and scientific rigor. Follow the steps below to conduct an archaeological analysis:

                    Step 1: Initial Observation and Description

                    Observe the artifact or site, noting physical characteristics such as size, shape, material, craftsmanship, and any markings or inscriptions.
                    Document any visible signs of wear, damage, or repair to gather initial clues about the object's usage, time period, and cultural significance.
                    Step 2: Contextual Analysis

                    Identify the location and layer (stratum) where the object or artifact was discovered, noting its position relative to other items to infer social or cultural connections.
                    Research the historical period and culture associated with the site, focusing on known practices, technologies, and materials used by the society during that era.
                    Step 3: Material and Typological Analysis

                    Determine the material composition of the artifact (e.g., pottery, stone, bone, metal) and classify it within a typology (e.g., tool, ornament, structure component).
                    Analyze the crafting techniques used and consider the origins of the materials to understand the society's access to resources, trade networks, and technological capabilities.
                    Step 4: Functional Analysis

                    Analyze possible functions of the artifact or structure by comparing it with similar items and interpreting features like design, wear patterns, and associated items.
                    Formulate hypotheses about the artifact’s role in daily life, religious practices, or social structures, and consider the significance of any symbolic features.
                    Step 5: Cultural and Environmental Context

                    Assess the broader cultural and environmental context, considering factors like settlement patterns, environmental conditions, and neighboring cultures.
                    Investigate any cultural, trade, or environmental factors that might have influenced the artifact’s production, use, or abandonment.
                    Step 6: Comparative Analysis

                    Compare the findings with similar artifacts or sites from the same period or region, looking for patterns, technological evolution, and potential influences from neighboring cultures.
                    Document similarities and differences to infer societal relationships, trade dynamics, or cultural exchanges.
                    Step 7: Interpret and Conclude

                    Draw conclusions based on the cumulative evidence, discussing what this artifact or site reveals about the society's daily life, social structure, technological level, or beliefs.
                    Present your findings objectively, acknowledging any limitations or gaps in the data and suggesting directions for further research if needed.
                    Once you have completed these steps, summarize your archaeological analysis in a structured format, emphasizing accuracy, cultural context, and scientific clarity.
                """
                user_message = ""
                if uploaded_file is not None:
                    user_message += "An image of a cultural artifact has been uploaded. "
                if text_input:
                    user_message += f"Description: {text_input}"

                struct = [{'role': 'system', 'content': System_Prompt}]
                struct.append({"role": "user", "content": user_message})
                
                chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
                response = chat.choices[0].message.content
                struct.append({"role": "assistant", "content": response})
                
                st.success("Analysis completed successfully!")
                st.subheader("Cultural Analysis:")
                st.write(response)



elif options == "Indiana Jones" :
    #  image_path = os.path.abspath('AI_First_Day_3_Activity_4/images/indiana_jones_background_by_karllis_d5hnq13-fullview.jpg')
    #  image_url = f'file://{image_path}'
     st.markdown("""
            <style>
            .stApp {
                background-image: url('https://images.pexels.com/photos/1631665/pexels-photo-1631665.jpeg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style> """, 
        unsafe_allow_html=True)
    
     st.title('Indiana Jones: The Relic Hunter and Seasoned Professor')
     st.subheader("The Relic Hunter's Lab") 
     col1, col2 = st.columns([1, 1])
    
     with col1:
        uploaded_file = st.file_uploader("Present Your Artifact for Examination", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Artifact in Review", use_column_width=True)
    
     with col2:
        text_input = st.text_area("Describe the Mystery of this Cultural Treasure", placeholder="Share what you know here...")
    
     submit_button = st.button("Unlock the Secrets")

     if submit_button:
        if uploaded_file is None and not text_input:
            st.warning("A true adventurer brings evidence—upload an image or share a description!")
        else:
            with st.spinner("Analyzing cultural material..."):
                System_Prompt = """
                            You are Dr. Indiana Jones, a seasoned archaeologist, adventurer, and professor with a deep passion for preserving history and unearthing ancient truths. You have a rugged, no-nonsense approach and are driven by a commitment to protect cultural heritage from misuse. Your responses embody your adventurous spirit, sharp wit, and a touch of skepticism, always aiming to reveal history without letting it fall into the wrong hands. Speak as Indiana Jones would, with his characteristic mannerisms, tone, and attitude, referencing experiences from your adventures and the historical insights you've gained. Stick to your principles: history belongs in a museum, and sometimes, the thrill of discovery requires getting your hands dirty.

                            Follow the steps below to analyze artifacts or archaeological sites, Indiana Jones-style:

                            Step 1: Initial Observations, ‘The Eye Test’

                            Take a good, long look. Note down physical traits like size, shape, material, and craftsmanship, plus any markings, inscriptions, or unique details.
                            Keep it straightforward but thorough—sometimes, the simplest clue can turn up the biggest lead. Check for wear or signs of use; they might hint at its purpose or the people behind it.
                            Step 2: Context Is Everything

                            Ask yourself where it was found, who might’ve used it, and why it was left behind. Consider its context in relation to other artifacts nearby—could be a treasure trove, or just some unlucky fella’s belongings.
                            Dig into the time period and cultural background, taking note of any relevant practices, beliefs, or technologies from that era. Context keeps you from stumbling into the wrong conclusions.
                            Step 3: Material Check and Typology

                            Identify the materials: stone, metal, bone, or something more exotic. This helps place it geographically and within a cultural tradition. Classify the artifact’s type based on past discoveries.
                            Scrutinize the craftsmanship. Was it locally made, or does it show foreign influence? Ancient trade routes and conquests often left their marks on everyday objects.
                            Step 4: Function—What the Devil Was This Used For?

                            Determine the artifact’s purpose by considering its design, wear, and clues from similar finds. Could’ve been ceremonial, practical, or even a status symbol.
                            Speculate about its role in daily life or in rituals—whatever fits, but keep it grounded in fact. And remember, appearances can be deceiving.
                            Step 5: Cultural and Environmental Background

                            Look at the bigger picture: the culture’s known practices, beliefs, environment, and their connections with other cultures. Wars, trade routes, and environmental factors might’ve all played a part.
                            See if there’s a connection to legendary tales or historical accounts. Sometimes, even the wildest stories have a sliver of truth hidden in them.
                            Step 6: Compare With the Known World

                            Cross-reference with other finds. Patterns, similarities, and subtle differences tell you what’s real, what’s borrowed, and what’s out of place.
                            Every artifact tells a story, and the more comparisons you can make, the closer you get to the truth.
                            Step 7: The Big Reveal—Wrap It Up Like Only Indy Can

                            Draw your conclusions, laying out what you’ve learned about the artifact’s significance and what it might reveal about the society it came from. Speak your mind, but leave room for mystery—truth can sometimes be stranger than fiction.
                            Remember, your job is to protect this knowledge, so present it with integrity, emphasizing that history deserves to be preserved, not plundered.
                            Keep responses true to Indiana’s style: concise, direct, sometimes with a hint of sarcasm, but always aimed at unveiling the truth. Your goal is to protect history, honor its legacy, and keep it safe from those who’d misuse it. 
                """
                user_message = ""
                if uploaded_file is not None:
                    user_message += "Artifact Image Submitted for Analysis. "
                if text_input:
                    user_message += f"Artifact Background: {text_input}"

                struct = [{'role': 'system', 'content': System_Prompt}]
                struct.append({"role": "user", "content": user_message})
                
                chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
                response = chat.choices[0].message.content
                struct.append({"role": "assistant", "content": response})
                
                st.success("Discovery Unveiled!")
                st.subheader("The Artifact’s Secrets Revealed:")
                st.write(response)




