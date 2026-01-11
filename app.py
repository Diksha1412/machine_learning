
import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# ------------------ NLTK SETUP ------------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ------------------ TEXT PREPROCESSING ------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ------------------ LOAD MODEL ------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ UI HEADER ------------------
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>📩 SMS / Email Spam Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Detect whether a message is <b>Spam</b> or <b>Not Spam</b> using AI 🤖</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------ INPUT ------------------
input_sms = st.text_area(
    "✉️ Enter your message below",
    height=150,
    placeholder="Congratulations! You won a free prize. Click now..."
)

# ------------------ PREDICT BUTTON ------------------
if st.button("🔍 Predict", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Probability
        prob = model.predict_proba(vector_input)[0]

        st.divider()

        # ------------------ RESULT ------------------
        col1, col2 = st.columns(2)

        with col1:
            if result == 1:
                st.error("🚨 SPAM MESSAGE")
            else:
                st.success("✅ NOT SPAM")

        with col2:
            st.metric("Spam Probability", f"{prob[1]*100:.2f}%")

        # ------------------ TEXT STATS ------------------
        st.info(
            f"""
            📊 **Message Stats**
            - Characters: {len(input_sms)}
            - Words: {len(input_sms.split())}
            """
        )

        # ------------------ SAVE HISTORY ------------------
        st.session_state.history.append({
            "Message": input_sms,
            "Prediction": "Spam" if result == 1 else "Not Spam",
            "Spam %": round(prob[1]*100, 2)
        })

# ------------------ HISTORY ------------------
if st.session_state.history:
    st.divider()
    st.subheader("📜 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("ℹ️ About App")
st.sidebar.markdown("""
**SMS / Email Spam Detection App**

**Tech Stack**
- Python 🐍
- Streamlit 🎈
- Scikit-learn 🤖
- NLTK 🧠

**ML Techniques**
- TF-IDF Vectorization
- Machine Learning Classifier
- NLP Text Processing

**Features**
- Real-time spam detection
- Confidence score
- Prediction history
- Clean UI
""")

