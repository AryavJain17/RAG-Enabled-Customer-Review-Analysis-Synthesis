import pandas as pd
import google.generativeai as genai
from retrieval import search_reviews

# === 1. Gemini API Config ===
genai.configure(api_key="AIzaSyBoc8G9zbFUiMRrj_h2xTBAFId0Q55x6so")
model = genai.GenerativeModel("gemini-1.5-flash")

# === 2. Load Metadata ===
df_meta = pd.read_csv("reviews_metadata.csv")

def calculate_sentiment_percentages(review_texts):
    matched_rows = df_meta[df_meta["reviews.text"].isin(review_texts)]
    if matched_rows.empty:
        return {"positive": 0, "negative": 0, "neutral": 0}
    total = len(matched_rows)
    positive = len(matched_rows[matched_rows["reviews.rating"] >= 4])
    negative = len(matched_rows[matched_rows["reviews.rating"] <= 2])
    neutral = total - positive - negative
    return {
        "positive": round((positive / total) * 100, 1),
        "negative": round((negative / total) * 100, 1),
        "neutral": round((neutral / total) * 100, 1)
    }

def answer_question(query):
    top_reviews = search_reviews(query, top_k=5)
    context = "\n".join(f"- {r}" for r in top_reviews)

    prompt = f"""
    You are an AI that summarizes customer reviews.
    Question: {query}
    Reviews:
    {context}

    Provide:
    1. Summary answer to the question.
    2. Sentiment analysis (positive/negative/neutral %).
    3. Key points from customers.
    """

    gemini_response = model.generate_content(prompt)
    local_sentiment = calculate_sentiment_percentages(top_reviews)

    return f"""
🔍 **Query:** {query}

--- Gemini AI Analysis ---
{gemini_response.text}

--- Verified Sentiment (from review ratings) ---
Positive: {local_sentiment['positive']}%
Negative: {local_sentiment['negative']}%
Neutral: {local_sentiment['neutral']}%
"""

def analyze_global_aspects(top_n=10):
    """Analyze most frequently mentioned aspects across ALL reviews."""
    all_reviews_text = "\n".join(f"- {txt}" for txt in df_meta["reviews.text"].tolist())

    prompt = f"""
    You are an AI that analyzes a large set of customer reviews.
    Reviews:
    {all_reviews_text}

    Task:
    Identify the top {top_n} most frequently mentioned aspects (features, issues, or qualities).
    For each aspect, provide:
    - Aspect name
    - Approximate mention frequency (percentage)
    - Overall sentiment (positive/negative/neutral)
    - Example customer quote
    Format as a numbered list.
    """

    gemini_response = model.generate_content(prompt)
    return f"""
📊 **Top {top_n} Most Frequently Mentioned Aspects in Entire Dataset**
{gemini_response.text}
"""

if __name__ == "__main__":
    print("💬 RAG-Enabled Customer Review Analysis (type 'exit' to quit, 'aspects' for global aspect analysis)\n")
    while True:
        query = input("Enter your question or command: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting...")
            break
        elif query.lower() == "aspects":
            print(analyze_global_aspects())
        else:
            print(answer_question(query))
