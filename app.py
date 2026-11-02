import json

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Customer Review Analyzer", page_icon="📊", layout="wide")

COLORS = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#f1c40f"}
BATCH_SIZE = 20          # reviews sent to the AI per request
SUMMARY_SAMPLE = 80      # reviews used for the summary / insights


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_models(api_key):
    """Ask Groq which chat models are currently available."""
    client = Groq(api_key=api_key)
    ids = [m.id for m in client.models.list().data]
    skip = ["whisper", "guard", "tts", "playai", "orpheus", "embed"]
    return sorted(i for i in ids if not any(word in i.lower() for word in skip))


def find_review_column(df):
    """Guess which column holds the review text."""
    keywords = ["review", "comment", "feedback", "text", "opinion", "description"]
    for col in df.columns:
        if any(k in str(col).lower() for k in keywords):
            return col
    # Otherwise pick the text column with the longest average length
    text_cols = df.select_dtypes(include="object").columns
    if len(text_cols) == 0:
        return None
    return max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())


def ask_groq_json(client, model, system_prompt, user_prompt):
    """Send a prompt to Groq and return the reply parsed as JSON."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def classify_batch(client, model, reviews):
    """Classify a small batch of reviews as Positive / Negative / Neutral."""
    numbered = "\n".join(f"{i}: {str(r)[:500]}" for i, r in enumerate(reviews))
    system_prompt = "You are a sentiment analysis expert. Reply only with valid JSON."
    user_prompt = (
        "Classify each numbered customer review as Positive, Negative or Neutral.\n"
        'Return JSON like: {"results": [{"id": 0, "sentiment": "Positive"}]}\n'
        "Include every review id.\n\n" + numbered
    )
    labels = ["Neutral"] * len(reviews)  # default if something goes wrong
    try:
        data = ask_groq_json(client, model, system_prompt, user_prompt)
        for item in data.get("results", []):
            idx = int(item["id"])
            label = str(item["sentiment"]).strip().capitalize()
            if 0 <= idx < len(reviews) and label in COLORS:
                labels[idx] = label
    except Exception:
        pass  # keep the default "Neutral" labels for this batch
    return labels


def generate_insights(client, model, reviews, counts):
    """Create the summary, complaints, positives and suggestions in one call."""
    sample = "\n".join(f"- {str(r)[:300]}" for r in reviews[:SUMMARY_SAMPLE])
    system_prompt = "You are a customer experience analyst. Reply only with valid JSON."
    user_prompt = (
        f"Sentiment counts: {counts}\n\n"
        f"Customer reviews:\n{sample}\n\n"
        "Return JSON with exactly these keys:\n"
        '"summary": a 4-6 sentence summary of all customer feedback,\n'
        '"satisfaction_level": one of "Very Low", "Low", "Moderate", "High", "Very High",\n'
        '"satisfaction_reason": one sentence explaining the level,\n'
        '"complaints": a list of up to 6 objects {"issue": "...", "details": "...", "frequency": "High/Medium/Low"},\n'
        '"liked_features": a list of up to 6 objects {"feature": "...", "details": "..."},\n'
        '"suggestions": a list of 5-7 practical improvement recommendations (strings).'
    )
    return ask_groq_json(client, model, system_prompt, user_prompt)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file of reviews", type=["csv"])

model = None
if api_key:
    try:
        models = get_available_models(api_key)
        if not models:
            st.sidebar.error("No chat models found for this API key.")
        else:
            default = models.index("llama-3.3-70b-versatile") if "llama-3.3-70b-versatile" in models else 0
            model = st.sidebar.selectbox("AI Model", models, index=default)
    except Exception:
        st.sidebar.error("❌ Invalid API key or could not connect to Groq.")

max_reviews = st.sidebar.slider("Maximum reviews to analyze", 10, 500, 100, step=10)
analyze_clicked = st.sidebar.button("🚀 Analyze Reviews", use_container_width=True)

# ---------------------------------------------------------------
# Main page
# ---------------------------------------------------------------
st.title("📊 AI Customer Review Analyzer")
st.caption("Upload customer reviews and get sentiment analysis, complaints, praise and suggestions.")

if uploaded_file is None:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.stop()

# ---- Read the CSV ----
df = None
last_error = None
for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
    try:
        uploaded_file.seek(0)  # go back to the start of the file before each try
        df = pd.read_csv(uploaded_file, encoding=encoding)
        break
    except Exception as e:
        last_error = e

if df is None:
    st.error(f"❌ Could not read the CSV file. Please upload a valid CSV. ({last_error})")
    st.stop()

if df.empty:
    st.error("❌ The CSV file is empty.")
    st.stop()

st.subheader("1. Data Preview")
st.dataframe(df.head(5), use_container_width=True)

# ---- Find the review column ----
guess = find_review_column(df)
if guess is None:
    st.error("❌ No text column found in this CSV. Please upload a file that contains reviews.")
    st.stop()

review_col = st.selectbox(
    "Review column (detected automatically — change it if wrong)",
    df.columns,
    index=list(df.columns).index(guess),
)

# ---- Run the analysis ----
if analyze_clicked:
    if not api_key:
        st.error("❌ Please enter your Groq API key in the sidebar.")
        st.stop()
    if model is None:
        st.error("❌ The API key is invalid, so the analysis cannot start.")
        st.stop()

    reviews = df[review_col].dropna().astype(str).str.strip()
    reviews = reviews[reviews != ""].head(max_reviews).tolist()
    if not reviews:
        st.error("❌ The selected column has no review text.")
        st.stop()

    client = Groq(api_key=api_key)
    try:
        with st.spinner("🤖 The AI is analyzing your reviews... please wait."):
            # Sentiment for every review (in batches)
            sentiments = []
            progress = st.progress(0.0)
            for start in range(0, len(reviews), BATCH_SIZE):
                batch = reviews[start:start + BATCH_SIZE]
                sentiments.extend(classify_batch(client, model, batch))
                progress.progress(min((start + BATCH_SIZE) / len(reviews), 1.0))
            progress.empty()

            result_df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
            counts = result_df["Sentiment"].value_counts().to_dict()

            # Summary, complaints, positives and suggestions
            insights = generate_insights(client, model, reviews, counts)

        st.session_state["result_df"] = result_df
        st.session_state["insights"] = insights
    except Exception as e:
        st.error(f"❌ Analysis failed. Check your API key and internet connection. ({e})")
        st.stop()

# ---------------------------------------------------------------
# Show results (kept in session_state so the page doesn't reset)
# ---------------------------------------------------------------
if "result_df" not in st.session_state:
    st.info("Click **Analyze Reviews** in the sidebar to start the analysis.")
    st.stop()

result_df = st.session_state["result_df"]
insights = st.session_state["insights"]

# ---- Metric cards ----
st.subheader("2. Dashboard")
total = len(result_df)
pos = int((result_df["Sentiment"] == "Positive").sum())
neg = int((result_df["Sentiment"] == "Negative").sum())
neu = int((result_df["Sentiment"] == "Neutral").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews", total)
c2.metric("😊 Positive", pos)
c3.metric("😡 Negative", neg)
c4.metric("😐 Neutral", neu)

# ---- Charts ----
chart_data = pd.DataFrame({"Sentiment": ["Positive", "Negative", "Neutral"], "Count": [pos, neg, neu]})
left, right = st.columns(2)
with left:
    pie = px.pie(chart_data, names="Sentiment", values="Count", color="Sentiment",
                 color_discrete_map=COLORS, title="Sentiment Share", hole=0.35)
    st.plotly_chart(pie, use_container_width=True)
with right:
    bar = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map=COLORS, title="Sentiment Distribution", text="Count")
    st.plotly_chart(bar, use_container_width=True)

# ---- Summary ----
st.subheader("3. Customer Feedback Summary")
st.write(insights.get("summary", "No summary available."))
st.success(f"**Overall satisfaction: {insights.get('satisfaction_level', 'N/A')}** — "
           f"{insights.get('satisfaction_reason', '')}")

# ---- Complaints and positives ----
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("4. 🚨 Common Complaints")
    for item in insights.get("complaints", []):
        st.error(f"**{item.get('issue', '')}** ({item.get('frequency', '')} frequency)\n\n{item.get('details', '')}")
with col_b:
    st.subheader("5. 👍 Most Appreciated Features")
    for item in insights.get("liked_features", []):
        st.success(f"**{item.get('feature', '')}**\n\n{item.get('details', '')}")

# ---- Suggestions ----
st.subheader("6. 💡 AI Suggestions")
for i, tip in enumerate(insights.get("suggestions", []), start=1):
    st.info(f"{i}. {tip}")

# ---- Sentiment table ----
st.subheader("7. Sentiment Results")
st.dataframe(result_df, use_container_width=True)

# ---- Download ----
st.download_button(
    label="⬇️ Download Analyzed Reviews (CSV)",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="analyzed_reviews.csv",
    mime="text/csv",
)
