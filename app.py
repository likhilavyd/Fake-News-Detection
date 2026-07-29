import os
import joblib
import streamlit as st
import plotly.express as px
import pandas as pd
from src.text_preprocessing import clean_text

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "best_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background:#0E1117;
}

.hero{
    background:linear-gradient(90deg,#1E3A8A,#2563EB);
    padding:35px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:25px;
}

.hero h1{
    font-size:42px;
    margin-bottom:5px;
}

.hero p{
    font-size:18px;
    opacity:0.9;
}

.real-card{
    background:#064E3B;
    color:white;
    padding:20px;
    border-radius:12px;
    font-size:28px;
    font-weight:bold;
    text-align:center;
}

.fake-card{
    background:#7F1D1D;
    color:white;
    padding:20px;
    border-radius:12px;
    font-size:28px;
    font-weight:bold;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
<h1>📰 Fake News Detection System</h1>
<p>
Detect whether a news article is <b>Fake</b> or <b>Real</b>
using Machine Learning, TF-IDF and Logistic Regression.
</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:

    st.title("📊 Model")

    st.success("Logistic Regression")

    st.write("Vectorizer")

    st.info("TF-IDF")

    st.divider()

    st.subheader("Performance")

    st.metric("Accuracy","98.96%")
    st.metric("Precision","99%")
    st.metric("Recall","99%")
    st.metric("F1 Score","99%")

    st.divider()

    st.caption("Developed using")

    st.write("• Python")
    st.write("• Scikit-Learn")
    st.write("• Streamlit")
    st.write("• Pandas")

left, right = st.columns([2,1])

with left:

    st.subheader("📝 Paste News Article")

    news = st.text_area(
        "",
        height=320,
        placeholder="Paste a complete news article here..."
    )

    col1, col2 = st.columns(2)

with col1:
    predict = st.button("🔍 Predict", use_container_width=True)

with col2:
    clear = st.button("🗑 Clear", use_container_width=True)

with right:

    st.subheader("ℹ About")

    st.write("""
This application classifies news articles using:

- TF-IDF
- Logistic Regression
- NLP preprocessing
- Lemmatization
- Stopword Removal
""")

if predict:

    if news.strip() == "":
        st.warning("Please paste a news article.")
    else:

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        fake = probability[0] * 100
        real = probability[1] * 100

        st.divider()

        if prediction == 0:

            st.markdown("""
            <div class="fake-card">
            🚨 FAKE NEWS
            </div>
            """, unsafe_allow_html=True)

            confidence = fake

        else:

            st.markdown("""
            <div class="real-card">
            ✅ REAL NEWS
            </div>
            """, unsafe_allow_html=True)

            confidence = real

            st.write("")

            c1, c2, c3 = st.columns(3)

            c1.metric("Confidence", f"{confidence:.2f}%")
            c2.metric("Fake", f"{fake:.2f}%")
            c3.metric("Real", f"{real:.2f}%")

            st.subheader("Prediction Probability")

            st.write("Fake News")

            st.progress(fake/100)

            st.write(f"{fake:.2f}%")

            st.write("")

            st.write("Real News")

            st.progress(real/100)

            st.write(f"{real:.2f}%")

            df = pd.DataFrame({
                "Class":["Fake","Real"],
                "Probability":[fake,real]
            })

            fig = px.bar(
                df,
                x="Class",
                y="Probability",
                text="Probability",
                color="Class",
                title="Prediction Confidence"
            )

            fig.update_layout(height=450)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

st.divider()

st.caption(
    "Built with Python • Scikit-Learn • Streamlit • TF-IDF • Logistic Regression"
)