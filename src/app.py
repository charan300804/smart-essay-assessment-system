import streamlit as st
import requests
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import textstat
from wordcloud import WordCloud
import spacy
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Fix for Streamlit import error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src import database

# Initialize Database
if 'db_initialized' not in st.session_state:
    database.init_db()
    st.session_state['db_initialized'] = True

# Load Spacy model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# Constants
API_HOST = os.getenv("API_HOST", "localhost")
API_URL = f"http://{API_HOST}:8000/score"

st.set_page_config(page_title="SEAS - Essay Dashboard", layout="wide", page_icon="🎓")

# --- Custom CSS ---
def local_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        /* Allow Streamlit's default background (Dark/Light) to take precedence or set a dark gradient */
        background: linear-gradient(to bottom right, #0e1117, #1a1c24); 
    }
    
    /* Headers - Force styling to pop against dark background */
    h1 {
        color: #ecf0f1 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.1);
        font-family: 'Segoe UI', sans-serif;
    }
    h2, h3 {
        color: #bdc3c7 !important;
    }
    
    /* Metric Cards - Glassmorphism */
    .metric-container {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.05); /* Transparent white */
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #4cd137 !important; /* Neon Green for numbers */
    }
    div[data-testid="stMetricLabel"] {
        color: #dcdde1 !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        margin: 5px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .badge-gold { background: linear-gradient(135deg, #f1c40f, #f39c12); color: black;}
    .badge-silver { background: linear-gradient(135deg, #bdc3c7, #95a5a6); color: black;}
    .badge-blue { background: linear-gradient(135deg, #3498db, #2980b9); }
    .badge-green { background: linear-gradient(135deg, #2ecc71, #27ae60); }
    .badge-purple { background: linear-gradient(135deg, #9b59b6, #8e44ad); }
    
    /* Input Areas */
    .stTextArea textarea {
        background-color: #1e272e !important;
        color: #f5f6fa !important;
        border: 1px solid #485460;
    }
    .stFileUploader {
        border: 2px dashed #00a8ff;
        background-color: rgba(0, 168, 255, 0.05);
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #dcdde1;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00a8ff;
        border-bottom-color: #00a8ff;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Helpers ---
def get_badges(structure, readability, score):
    badges = []
    if score >= 9:
        badges.append(("🏆 Top Scorer", "badge-gold"))
    elif score >= 7:
        badges.append(("🥈 High Achiever", "badge-silver"))
        
    if structure['Word Count'] > 300:
        badges.append(("✍️ Prolific Writer", "badge-blue"))
        
    if readability['Flesch-Kincaid Grade'] > 12:
        badges.append(("🎓 Scholar", "badge-purple"))
    elif readability['Flesch-Kincaid Grade'] >= 8:
        badges.append(("🏫 High School Hero", "badge-green"))
        
    avg_len = structure['Avg. Sentence Length']
    if avg_len > 15 and avg_len < 25:
        badges.append(("🌊 Fluid Flow", "badge-blue"))
        
    return badges

def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        number = {'valueformat': '.2f'},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Essay Score"},
        gauge = {
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2ecc71" if score > 7 else "#f1c40f" if score > 5 else "#e74c3c"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': '#ffebee'},
                {'range': [5, 8], 'color': '#fff3e0'},
                {'range': [8, 10], 'color': '#e8f5e9'}],
        }))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
    return fig

# --- Main App ---
# History is now managed by database.py
if 'history' not in st.session_state:
    st.session_state.history = [] # Keep for temporary state if needed, but primarily use DB

def main():
    col_logo, col_title = st.columns([1, 5])
    with col_title:
        st.title("SEAS: Intelligent Essay Dashboard")
        st.caption("Advanced Scoring • Deep Analytics • Interactive Feedback")

    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        if st.button("➕ New Analysis", use_container_width=True):
            st.session_state.pop('result', None)
            st.rerun()
        
        if st.button("🗑️ Clear History"):
            database.clear_history()
            st.rerun()
            
        st.header("🕒 Recent Activity")
        
        # Load History from DB
        db_history = database.get_history()
        
        if db_history:
            for entry in db_history[:5]: # Show last 5
                with st.container():
                    st.markdown(f"**Score: {entry['score']:.2f}**")
                    st.caption(f"{entry['time']} • {entry['words']} words")
                    st.divider()
        else:
            st.info("No essays analyzed yet.")
            
        st.markdown("### 🏆 Your Badges")
        
        # Calculate Badge Counts from DB History
        badge_counts = Counter()
        for entry in db_history:
             if 'badges' in entry:
                for badge_name, _ in entry['badges']:
                     if badge_name: # Ensure not empty string
                        badge_counts[badge_name] += 1
        
        if badge_counts:
            for badge, count in badge_counts.items():
                # specific styles for known badges
                style = "badge-blue"
                if "Top" in badge: style = "badge-gold"
                elif "High" in badge: style = "badge-silver"
                elif "Scholar" in badge: style = "badge-purple"
                elif "Hero" in badge: style = "badge-green"
                
                st.markdown(f'<span class="badge {style}">{badge} x{count}</span>', unsafe_allow_html=True)
        else:
             st.caption("Analyze essays to earn badges!")

    # --- Input Section ---
    c1, c2 = st.columns([1, 1])
    with c1:
        uploaded_file = st.file_uploader("📂 Upload Essay (.txt)", type="txt")
    with c2:
        st.info("💡 Tip: Essays between 300-600 words get the most accurate structural analysis.")

    default_text = ""
    # Check if a file was just uploaded to overwrite
    if uploaded_file:
        default_text = uploaded_file.read().decode("utf-8")
    
    # Use a shorter height for cleaner look
    essay_text = st.text_area("✍️ Compose your essay:", value=default_text, height=250, placeholder="Start typing here...")

    if st.button("🚀 Analyze Essay", use_container_width=True):
        if not essay_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("🤖 AI is reading your essay..."):
                try:
                    # NLP Analysis (Fastest first)
                    doc = nlp(essay_text)
                    pos_counts = Counter([token.pos_ for token in doc])
                    
                    structure = {
                        "Word Count": len(doc),
                        "Sentence Count": len(list(doc.sents)),
                        "Avg. Sentence Length": np.mean([len(sent) for sent in doc.sents]) if len(list(doc.sents)) > 0 else 0
                    }
                    
                    readability = {
                        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(essay_text),
                        "Reading Ease": textstat.flesch_reading_ease(essay_text)
                    }

                    # API Call
                    payload = {"essay": essay_text}
                    response = requests.post(API_URL, json=payload)
                    score = 0.0
                    if response.status_code == 200:
                        score = response.json().get("score", 0.0)
                        
                    # Calculate Badges
                    current_badges = get_badges(structure, readability, score)

                    # Store in Database
                    database.add_essay(
                        score=score, 
                        word_count=structure["Word Count"], 
                        badges=current_badges,
                        snippet=essay_text[:50]
                    )
                    
                    # Store Result
                    st.session_state['result'] = {
                        "score": score,
                        "text": essay_text,
                        "pos": pos_counts,
                        "structure": structure,
                        "readability": readability,
                        "badges": current_badges
                    }
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # --- Results Dashboard ---
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        st.markdown("### 📊 Assessment Report")
        
        # Row 1: Score Gauge & Key Metrics
        row1_1, row1_2 = st.columns([1, 2])
        
        with row1_1:
            st.plotly_chart(create_gauge(res['score']), use_container_width=True)
            
        with row1_2:
            st.markdown("#### Achievements Unlocked")
            if res['badges']:
                badges_html = " ".join([f'<span class="badge {b[1]}">{b[0]}</span>' for b in res['badges']])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.caption("No specific badges unlocked this time. Keep trying!")
            
            st.divider()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Grade Level", f"{res['readability']['Flesch-Kincaid Grade']:.2f}")
            m2.metric("Reading Ease", f"{res['readability']['Reading Ease']:.2f}")
            m3.metric("Avg Sent Length", f"{res['structure']['Avg. Sentence Length']:.1f}")

        # Row 2: Deep Dive Tabs
        tab_Analysis, tab_Vis, tab_Feedback = st.tabs(["📝 Text Analysis", "📈 Visual Insights", "💡 Feedback"])
        
        with tab_Analysis:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("**Part of Speech Breakdown**")
                # Interactive Bar Chart
                pos_df = pd.DataFrame.from_dict(res['pos'], orient='index', columns=['Count']).reset_index()
                fig_pos = px.bar(pos_df, x='index', y='Count', labels={'index': 'Type', 'Count': 'Frequency'}, color='Count')
                st.plotly_chart(fig_pos, use_container_width=True)
            
            with col_a2:
                st.markdown("**Essay Statistics**")
                st.json(res['structure'])

        with tab_Vis:
            st.markdown("**Vocabulary Cloud**")
            wordcloud = WordCloud(width=800, height=400, background_color='#f8f9fa', colormap='viridis').generate(res['text'])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
        with tab_Feedback:
            st.subheader("Improvement Tips")
            if res['score'] < 6:
                st.warning("❌ Your essay needs more structure. Try organizing your thoughts into clear paragraphs.")
            if res['readability']['Reading Ease'] < 30:
                st.info("ℹ️ Your writing is quite complex. Consider simplifying sentences for broader appeal.")
            if res['structure']['Avg. Sentence Length'] < 10:
                st.info("ℹ️ Sentences are short. Try combining ideas with conjunctions.")
            if not res['badges']:
                 st.info("💡 Try to increase your word count and vocabulary variety to unlock badges!")
            
            st.success("✅ Good use of vocabulary!" if len(res['pos']) > 10 else "⚠️ Try varying your vocabulary more.")

if __name__ == "__main__":
    main()
