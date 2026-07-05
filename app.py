# -----------------------------
# 📦 Import Required Libraries
# -----------------------------
import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------
# 🔧 Download NLTK Resources 
# -----------------------------
@st.cache_resource(show_spinner=False)
def download_nltk_resources():
    """Download required NLTK resources silently."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        st.error(f"❌ Failed to download NLTK resources: {e}")
        return False

# Initialize NLTK resources
download_nltk_resources()

# -----------------------------
# ⚙️ Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 🎨 Load External CSS File
# -----------------------------
def load_css(file_path: str) -> None:
    """
    Loads external CSS file into the Streamlit app.
    
    Args:
        file_path (str): Path to CSS file
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ CSS file not found at: {file_path}")
    except Exception as e:
        st.error(f"❌ Error loading CSS: {e}")

# Load the CSS from assets folder
load_css("style.css")

# -----------------------------
# 📦 Load Model & Vectorizer (Cached)
# -----------------------------
@st.cache_resource(show_spinner="🔄 Loading AI Model...")
def load_model_and_vectorizer():
    """
    Loads the trained Logistic Regression model and TF-IDF vectorizer.
    Uses @st.cache_resource so it loads only ONCE for the entire session.
    
    Returns:
        tuple: (model, vectorizer) or (None, None) on failure
    """
    try:
        # Load Logistic Regression Model
        with open("fake_news_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        
        # Load TF-IDF Vectorizer
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        
        return model, vectorizer
    
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None, None
    except pickle.UnpicklingError as e:
        st.error(f"❌ Failed to unpickle model file: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {e}")
        return None, None

# Load model and vectorizer once at startup
model, vectorizer = load_model_and_vectorizer()

# -----------------------------
# 🧹 Text Cleaning Function
# -----------------------------
# Initialize outside function for performance (loaded once)
STEMMER = PorterStemmer()
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))

@st.cache_data(show_spinner=False)
def clean_text(text: str) -> str:
    """
    Cleans and preprocesses input text - EXACTLY same as training pipeline
    (see Notebook/Fake_news_Detector.ipynb). This must stay in sync with the
    notebook's clean_text() because the TF-IDF vectorizer's vocabulary was
    built on text cleaned this exact way. Adding/removing steps here (e.g.
    stripping URLs or digits) would feed the model text shaped differently
    from what it was trained on and quietly hurt prediction quality.
    
    Pipeline Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Tokenization (NLTK word_tokenize)
        4. Remove stopwords
        5. Apply Porter Stemming
    
    Args:
        text (str): Raw news article text
    
    Returns:
        str: Cleaned and preprocessed text
    """
    try:
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Remove punctuation
        for p in string.punctuation:
            text = text.replace(p, "")
        
        # Step 3: Tokenization (NLTK word_tokenize, same as training)
        tokens = word_tokenize(text)
        
        # Step 4: Remove stopwords
        tokens = [word for word in tokens if word not in STOP_WORDS]
        
        # Step 5: Apply Porter Stemming
        cleaned_tokens = [STEMMER.stem(word) for word in tokens]
        
        # Step 6: Join tokens back into a single string
        return " ".join(cleaned_tokens)
    
    except Exception as e:
        st.error(f"❌ Error cleaning text: {e}")
        return ""

# -----------------------------
# 🔮 Prediction Function
# -----------------------------
def predict_news(news_text: str):
    """
    Predicts whether given news is Fake or Real.
    
    Args:
        news_text (str): Raw news article text
    
    Returns:
        dict: Contains prediction, confidence, and probabilities
    """
    try:
        # Step 1: Clean the text
        cleaned = clean_text(news_text)
        
        if not cleaned:
            return {"error": "Text cleaning resulted in empty content."}
        
        # Step 2: Convert cleaned text using TF-IDF
        vectorized_text = vectorizer.transform([cleaned])
        
        # Step 3: Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Step 4: Get prediction probabilities
        probabilities = model.predict_proba(vectorized_text)[0]
        
        # Step 5: Extract confidence score for predicted class
        confidence = float(probabilities[prediction]) * 100
        
        # Step 6: Return structured result
        return {
            "prediction": int(prediction),
            "confidence": confidence,
            "fake_prob": float(probabilities[0]) * 100,
            "real_prob": float(probabilities[1]) * 100,
            "error": None
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# -----------------------------
# 📋 Sidebar - Project Information
# -----------------------------
with st.sidebar:
    st.title("🤖 About Project")
    
    st.markdown("""
    ### Fake News Detection AI
    
    **🧠 Model:** Logistic Regression
    
    **📊 Vectorizer:** TF-IDF
    
    **🎯 Accuracy:** 97%
    
    **📚 Dataset:** WELFake Dataset
    
    **📈 Records:** 45,757
    
    ---
    
    ### 🛠️ Tech Stack
    - Python 3.11
    - Streamlit
    - Scikit-learn
    - NLTK
    - Pandas & NumPy
    
    ---
    
    ### 📌 How It Works
    1. Paste news article
    2. Click Analyze
    3. Get instant prediction
    4. View confidence score
    
    ---
    
    👨‍💻 Developed by **Nadir Khan**
    """)

# -----------------------------
# 🎯 Main Application Header
# -----------------------------
st.title("📰 Fake News Detection AI")

st.markdown("""
Detect whether a news article is **Real** or **Fake** using
Natural Language Processing (NLP) and Machine Learning.
""")

st.divider()

# -----------------------------
# 📝 News Input Section
# -----------------------------
news = st.text_area(
    "📝 Paste News Article",
    height=250,
    placeholder="Paste your news article here... (Minimum 20 characters recommended)",
    help="Paste the complete news article for best accuracy."
)

# Display character count
char_count = len(news.strip())
st.caption(f"📊 Character count: **{char_count}**")

# -----------------------------
# 🔍 Analyze Button
# -----------------------------
predict = st.button(
    "🔍 Analyze News",
    use_container_width=True,
    type="primary"
)

st.divider()

# -----------------------------
# 🎯 Prediction Logic & Results
# -----------------------------
if predict:
    
    # Validation 1: Check if model loaded successfully
    if model is None or vectorizer is None:
        st.error("❌ Model or Vectorizer not loaded. Please check your .pkl files.")
    
    # Validation 2: Check for empty input
    elif news.strip() == "":
        st.warning("⚠️ Please enter a news article to analyze.")
    
    # Validation 3: Check for minimum text length
    elif len(news.strip()) < 20:
        st.warning("⚠️ Text is too short. Please enter at least 20 characters for accurate prediction.")
    
    # All good - proceed with prediction
    else:
        # Show loading spinner while predicting
        with st.spinner("🔎 Analyzing news article... Please wait"):
            result = predict_news(news)
        
        # Handle prediction errors
        if result.get("error"):
            st.error(f"❌ {result['error']}")
        
        else:
            # Extract results
            prediction = result["prediction"]
            confidence = result["confidence"]
            fake_prob = result["fake_prob"]
            real_prob = result["real_prob"]
            
            # -----------------------------
            # 🎨 Display Result Card
            # -----------------------------
            if prediction == 1:
                # REAL NEWS
                st.markdown(f"""
                <div class="result-card real-news">
                    <div class="result-icon">✅</div>
                    <div class="result-title">REAL NEWS</div>
                    <div class="result-subtitle">This article appears to be authentic and trustworthy.</div>
                    <div class="confidence-text">Confidence: <strong>{confidence:.2f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # FAKE NEWS
                st.markdown(f"""
                <div class="result-card fake-news">
                    <div class="result-icon">🚨</div>
                    <div class="result-title">FAKE NEWS</div>
                    <div class="result-subtitle">This article appears to be misleading or false.</div>
                    <div class="confidence-text">Confidence: <strong>{confidence:.2f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            
            # -----------------------------
            # 📊 Confidence Progress Bar
            # -----------------------------
            st.markdown("### 📊 Confidence Level")
            st.progress(round(confidence), text=f"{confidence:.2f}% Confident")
            
            # -----------------------------
            # 📈 Detailed Probability Breakdown
            # -----------------------------
            st.markdown("### 📈 Probability Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🚨 Fake News Probability",
                    value=f"{fake_prob:.2f}%",
                    delta=f"{fake_prob - 50:.2f}%" if fake_prob > 50 else None,
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    label="✅ Real News Probability",
                    value=f"{real_prob:.2f}%",
                    delta=f"{real_prob - 50:.2f}%" if real_prob > 50 else None,
                    delta_color="normal"
                )
            
            # -----------------------------
            # ℹ️ Additional Information
            # -----------------------------
            with st.expander("ℹ️ How was this prediction made?"):
                st.markdown(f"""
                **Prediction Pipeline:**
                
                1. **Text Cleaning** - Removed punctuation, URLs, numbers, and stopwords
                2. **Stemming** - Reduced words to their root form
                3. **TF-IDF Vectorization** - Converted text into numerical features
                4. **Logistic Regression** - Classified as Real or Fake
                
                **Model Details:**
                - Trained on **45,757** news articles (WELFake Dataset)
                - Achieved **97% accuracy** on test data
                - Uses TF-IDF features for text representation
                
                **⚠️ Disclaimer:** This is an AI prediction based on patterns in training data.
                Always verify news from multiple trusted sources.
                """)

# -----------------------------
# 🔻 Custom Footer
# -----------------------------
st.markdown("""
<div class="custom-footer">
    © 2026 Fake News Detection AI | Built with ❤️ using Streamlit & Machine Learning<br>
    Developed by <strong>Nadir Khan</strong>
</div>  
""", unsafe_allow_html=True)