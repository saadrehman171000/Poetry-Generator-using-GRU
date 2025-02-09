import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
import json
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cohere
from dotenv import load_dotenv
import plotly.graph_objects as go
from collections import Counter
import re
import time

# Load environment variables
load_dotenv()

# Initialize Cohere client securely
co = cohere.Client(os.getenv('COHERE_API_KEY', st.secrets["COHERE_API_KEY"]))
def explain_poetry_for_everyone(poetry):
    try:
        prompt = f"""
        Analyze this Roman Urdu poetry and explain it in simple terms:

        {poetry}

        Explain in these sections:

        Main Message:
        [Explain the core meaning simply]

        Hidden Meanings:
        [Explain any metaphors or deeper meanings]

        Emotions:
        [Describe the feelings expressed]

        Modern Example:
        [Give a relatable real-life example]

        Cultural Context:
        [Share an interesting cultural or literary insight]

        Format the response in clear paragraphs, not as JSON.
        """
        
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            return_likelihoods='NONE'
        )
        
        explanation = response.generations[0].text.strip()
        
        # Format the sections with emojis
        formatted_explanation = explanation.replace("Main Message:", "🎯 Main Message:")
        formatted_explanation = formatted_explanation.replace("Hidden Meanings:", "\n💭 Hidden Meanings:")
        formatted_explanation = formatted_explanation.replace("Emotions:", "\n❤️ Emotions:")
        formatted_explanation = formatted_explanation.replace("Modern Example:", "\n🌟 Modern Example:")
        formatted_explanation = formatted_explanation.replace("Cultural Context:", "\n🏺 Cultural Context:")
        
        return formatted_explanation
    except Exception as e:
        st.error(f"Error explaining poetry: {str(e)}")
        return None

def analyze_poetry_mood(poetry):
    try:
        # Define emotion keywords
        emotion_keywords = {
            'Love': ['dil', 'mohabbat', 'ishq', 'pyar', 'yaad'],
            'Sadness': ['dard', 'gham', 'judai', 'tanhai', 'aansu'],
            'Hope': ['umeed', 'khwab', 'armaan', 'tamanna', 'chahat'],
            'Nature': ['chand', 'sitare', 'phool', 'hawa', 'barish'],
            'Life': ['zindagi', 'waqt', 'safar', 'duniya', 'manzil']
        }
        
        # Count emotion occurrences
        emotion_scores = {emotion: 0 for emotion in emotion_keywords}
        words = poetry.lower().split()
        
        for word in words:
            for emotion, keywords in emotion_keywords.items():
                if word in keywords:
                    emotion_scores[emotion] += 1
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(emotion_scores.values()),
            theta=list(emotion_scores.keys()),
            fill='toself',
            name='Emotional Intensity'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(emotion_scores.values()) + 1]
                )),
            showlegend=False,
            title="Poetry Mood Analysis"
        )
        
        return fig, emotion_scores
    except Exception as e:
        st.error(f"Error analyzing mood: {str(e)}")
        return None, None

def get_dominant_emotions(emotion_scores):
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    return [emotion for emotion, score in sorted_emotions if score > 0][:2]

# Define Poetry Styles
POETRY_STYLES = {
    "Love": ["mohabbat ki kahani", "teri yaad", "ishq ke rang", "dil ke armaan"],
    "Nature": ["chandni raat", "khushbu hawa", "barish ki boondein", "phool khile"],
    "Philosophy": ["zindagi ke safar", "waqt ki raftar", "khwabon ki duniya", "soch ka darya"],
    "Sadness": ["tanhai ke lamhe", "dard ke saaye", "judai ka gham", "aansu behte"]
}

# Update the dictionary with more words and common phrases
URDU_ENGLISH_DICT = {
    # Love related words
    'mohabbat': 'love',
    'ki': 'of',
    'kahani': 'story',
    'suud': 'profit',
    'duub': 'sunset',
    
    # Time and life related
    'din': 'day',
    'jidhar': 'where',
    'meri': 'my',
    'zindagi': 'life',
    'hi': 'only',
    'meñ': 'in',
    'vo': 'that',
    'yuñhī': 'just like this',
    
    # Emotional words
    'baal': 'hair',
    'haiñ': 'are',
    'shab': 'night',
    'e': 'of',
    'furqat': 'separation',
    'shāyad': 'perhaps',
    
    # State of being
    'bane': 'became',
    'kitne': 'how many',
    'rusvā': 'disgraced',
    'hoñ': 'am',
    
    # Nature and conditions
    'badle': 'changed',
    'zinda': 'alive',
    'tabī.at': 'nature',
    'raho': 'stay',
    'jo': 'who/which',
    'sab': 'all',
    
    # Common words from previous examples
    'dil': 'heart',
    'ke': 'of',
    'armaan': 'desires',
    'namnāk': 'moist',
    'kyuuñ': 'why',
    'fughāñ': 'lament',
    'bīmār': 'sick',
    'nisār': 'sacrifice',
    'kal': 'tomorrow',
    'bahut': 'very',
    'hotī': 'happens',
    'hai': 'is',
    'har': 'every',
    'be-ikhtiyār': 'helpless',
    'ham': 'we',
    'ne': 'have'
}

def get_style_suggestions(selected_styles):
    suggestions = []
    for style in selected_styles:
        if style in POETRY_STYLES:
            suggestions.extend(POETRY_STYLES[style])
    return suggestions if suggestions else ["mohabbat ki kahani"]

# Page config
st.set_page_config(
    page_title="✨ Urdu Poetry Generator",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean, modern CSS with light theme
st.markdown("""
    <style>
    /* Light Theme Base */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Header */
    .header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    
    .header h1 {
        color: #1a1a1a;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Input Fields */
    .stTextInput input {
        background: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        color: #1a1a1a;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #ff4b82;
        box-shadow: 0 0 0 4px rgba(255,75,130,0.1);
    }
    
    /* Sliders */
    .stSlider div[data-baseweb="slider"] {
        background: #e9ecef;
    }
    
    .stSlider div[role="slider"] {
        background: #ff4b82;
        box-shadow: 0 2px 6px rgba(255,75,130,0.2);
    }
    
    /* Generate Button */
    .stButton button {
        background: linear-gradient(45deg, #ff4b82, #ff758c);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255,75,130,0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255,75,130,0.3);
    }
    
    /* Poetry Output */
    .poetry-output {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        line-height: 2;
        color: #1a1a1a;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        gap: 4px;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #666;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,75,130,0.05);
        color: #ff4b82;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ff4b82 !important;
        color: white !important;
    }
    
    /* Section Headers */
    h3 {
        color: #1a1a1a;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Download Button */
    .download-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #f8f9fa;
        color: #ff4b82;
        padding: 8px 16px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .download-button:hover {
        background: #ff4b82;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Success Message */
    .stSuccess {
        background: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
        padding: 1rem;
        border-radius: 6px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
    
    .footer a {
        color: #ff4b82;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        color: #1a1a1a;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ff4b82;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666;
    }
    
    /* Translation Output */
    .translation-output {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .translation-output:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .translation-output h4 {
        color: #ff4b82;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <div style="background: #ff4b82; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">✨</div>
        <h1>Urdu Poetry Generator</h1>
    </div>
    <p style="text-align: center; color: #666; margin-bottom: 2rem; font-size: 1.1rem;">
        Create beautiful Urdu poetry using artificial intelligence
    </p>
""", unsafe_allow_html=True)

# Define utility functions
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'

def load_history():
    try:
        with open('poetry_history.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        # Create file if it doesn't exist
        if not os.path.exists('poetry_history.json'):
            with open('poetry_history.json', 'w', encoding='utf-8') as f:
                json.dump([], f)
        return []

def save_to_history(poetry, prompt):
    history = load_history()
    history.append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prompt': prompt,
        'poetry': poetry
    })
    with open('poetry_history.json', 'w', encoding='utf-8') as f:
        json.dump(history[-50:], f)  # Keep last 50 entries

# Load the model and encoder
@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model("poetry_gru_model.h5")
        
        with open("word_encoder.pkl", "rb") as f:
            word_encoder = pickle.load(f)
            
        word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
        index_to_word = {i: word for word, i in word_to_index.items()}
        
        return model, word_to_index, index_to_word
    except Exception as e:
        st.error(f"Error loading model or encoder: {str(e)}")
        return None, None, None

def generate_nazam(start_text, words_per_line, total_lines, model, word_to_index, index_to_word):
    try:
        generated_text = start_text.split()
        
        for _ in range(total_lines * words_per_line):
            encoded_input = [word_to_index.get(word, 0) for word in generated_text[-5:]]
            encoded_input = pad_sequences([encoded_input], maxlen=5, truncating="pre")
            
            predicted_probs = model.predict(encoded_input, verbose=0)
            predicted_index = np.argmax(predicted_probs, axis=-1)[0]
            next_word = index_to_word.get(predicted_index, "")
            
            if not next_word:
                continue
                
            generated_text.append(next_word)
            
            if len(generated_text) % words_per_line == 0:
                generated_text.append("\n")
                
        return " ".join(generated_text)
    except Exception as e:
        st.error(f"🚫 Error generating poetry: {str(e)}")
        return ""

# Load model and check
model, word_to_index, index_to_word = load_model_and_encoder()

if not all([model, word_to_index, index_to_word]):
    st.error("⚠️ Failed to load required components. Please check if all files exist.")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🎨 Generate", "📚 History"])

# Add this after imports
if 'generated_poetry' not in st.session_state:
    st.session_state.generated_poetry = None

# Function to handle poetry generation
def generate_and_store_poetry():
    poetry = generate_nazam(
        st.session_state.start_text,
        st.session_state.words_per_line,
        st.session_state.total_lines,
        model, word_to_index, index_to_word
    )
    if poetry:
        st.session_state.generated_poetry = poetry
        save_to_history(poetry, st.session_state.start_text)

# Update the main tab content
with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        selected_styles = st.multiselect(
            "Select Poetry Styles to Mix",
            list(POETRY_STYLES.keys()),
            default=["Love"],
            key="style_mixer"
        )
        
        if selected_styles:
            suggestions = get_style_suggestions(selected_styles)
            selected_suggestion = st.selectbox(
                "💡 Suggested Starting Words",
                suggestions,
                key="suggestion_selector"
            )
            st.session_state.start_text = st.text_input(
                "Starting Words",
                value=selected_suggestion,
                key="poetry_input"
            )
        else:
            st.session_state.start_text = st.text_input(
                "Starting Words",
                value="mohabbat ki kahani",
                key="poetry_input"
            )

    with col2:
        st.markdown("### ⚙️ Parameters")
        st.session_state.words_per_line = st.slider(
            "Words per Line",
            3, 15, 5,
            key="words_slider"
        )
        st.session_state.total_lines = st.slider(
            "Total Lines",
            2, 10, 5,
            key="lines_slider"
        )

    # Generate Button
    if st.button("✨ Generate Poetry", key="generate_btn", use_container_width=True):
        with st.spinner("Creating your masterpiece..."):
            generate_and_store_poetry()

    # Display poetry
    if st.session_state.generated_poetry:
        st.markdown(f"""
            <div class="poetry-output">
                {st.session_state.generated_poetry.replace('\n', '<br>')}
            </div>
        """, unsafe_allow_html=True)

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(get_download_link(
                st.session_state.generated_poetry,
                f"poetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "⬇️ Download Poetry"
            ), unsafe_allow_html=True)
        with col2:
            if st.button("📋 Copy", key="copy_btn"):
                st.success("✨ Copied to clipboard!")

        # Poetry Explanation
        with st.expander("✨ Poetry Explained"):
            with st.spinner("🤔 Understanding the poetry's deeper meaning..."):
                explanation = explain_poetry_for_everyone(st.session_state.generated_poetry)
                if explanation:
                    # Add some styling
                    st.markdown("""
                        <style>
                        .poetry-explanation {
                            padding: 20px;
                            border-radius: 10px;
                            background-color: #f8f9fa;
                            margin: 10px 0;
                        }
                        .poetry-explanation h3 {
                            color: #ff4b82;
                            margin-bottom: 15px;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="poetry-explanation">
                            {explanation}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Interactive feedback
                    st.markdown("---")
                    st.markdown("### Was this explanation helpful?")
                    col1, col2, _ = st.columns([1, 1, 2])
                    with col1:
                        if st.button("👍 Yes", key="helpful_yes"):
                            st.balloons()
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎 No", key="helpful_no"):
                            st.info("Thanks! We'll try to improve our explanations.")

with tab2:
    # History tab content
    st.subheader("📚 Generation History")
    history = load_history()
    if history:
        for idx, entry in enumerate(reversed(history)):
            with st.expander(f"🕒 {entry['date']} - Prompt: {entry['prompt'][:30]}..."):
                # Create three columns for better organization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📝 Poetry")
                    st.text_area(
                        "Poetry", 
                        entry['poetry'], 
                        height=150,
                        key=f"history_poetry_{idx}"
                    )
                    st.markdown(get_download_link(
                        entry['poetry'],
                        f"poetry_{entry['date'].replace(' ', '_')}.txt",
                        "📥 Download"
                    ), unsafe_allow_html=True)
                    
                    # Add explanation directly below poetry
                    st.markdown("### ✨ Poetry Explanation")
                    with st.spinner("Analyzing poetry..."):
                        explanation = explain_poetry_for_everyone(entry['poetry'])
                        if explanation:
                            st.markdown(explanation)
                
                with col2:
                    st.markdown("### 📊 Analysis")
                    
                    # Mood Analysis
                    fig, emotion_scores = analyze_poetry_mood(entry['poetry'])
                    if fig and emotion_scores:
                        # Show dominant emotions
                        dominant_emotions = get_dominant_emotions(emotion_scores)
                        if dominant_emotions:
                            st.markdown("#### 🎭 Dominant Emotions")
                            for emotion in dominant_emotions:
                                st.markdown(f"- {emotion}")
                        
                        # Display mood radar chart
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Word Statistics
                    st.markdown("#### 📈 Statistics")
                    words = entry['poetry'].split()
                    unique_words = len(set(words))
                    
                    # Create metrics
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("Total Words", len(words))
                    with stat_col2:
                        st.metric("Unique Words", unique_words)
                    
                    # Word frequency
                    word_freq = Counter(words)
                    most_common = word_freq.most_common(3)
                    
                    st.markdown("#### 🔤 Most Used Words")
                    for word, count in most_common:
                        st.markdown(f"- **{word}**: {count} times")
    else:
        st.info("No generation history yet. Create some poetry first!")

# About Section
with st.expander("ℹ️ About this Poetry Generator"):
    st.markdown("""
    ### How it Works
    This poetry generator uses a sophisticated Gated Recurrent Unit (GRU) neural network 
    trained on a vast collection of Roman Urdu poetry. The model learns patterns and 
    structures from existing poetry to generate new, unique verses.

    ### Tips for Best Results
    1. Start with meaningful Roman Urdu words
    2. Try different combinations of words per line
    3. Adjust total lines to get shorter or longer poems
    4. Experiment with different starting phrases

    ### Technical Details
    - Model: GRU (Gated Recurrent Unit)
    - Training Data: Roman Urdu Poetry Collection
    - Output: Generated poetry in Roman Urdu script
    """)

# Footer
st.markdown("""
---
<p style='text-align: center; color: #666;'>
    Made with ❤️ for Urdu Poetry | 
    <a href="https://github.com/saadrehman171000/Poetry-Generator-using-GRU" target="_blank">GitHub</a>
</p>
""", unsafe_allow_html=True)