import pandas as pd
import re
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory

# Fix random langdetect results
DetectorFactory.seed = 0

# ==== Step 1: Load dataset ====
df = pd.read_csv("dataset.csv")

# ==== Step 2: Keep only relevant columns ====
df = df[[
    "name",
    "brand",
    "categories",
    "reviews.date",
    "reviews.rating",
    "reviews.text",
    "reviews.username"
]]

# ==== Step 3: Drop rows with missing review text ====
df = df.dropna(subset=["reviews.text"])

# ==== Step 4: Remove duplicates ====
df = df.drop_duplicates(subset=["reviews.text"])

# ==== Step 5: Clean review text ====
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(str(text), "html.parser").get_text()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-ASCII chars (optional)
    text = text.encode("ascii", errors="ignore").decode()
    return text.strip()

df["reviews.text"] = df["reviews.text"].apply(clean_text)

# ==== Step 6: Remove very short reviews ====
df = df[df["reviews.text"].str.split().str.len() >= 5]

# ==== Step 7 (Optional): Keep only English reviews ====
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

df = df[df["reviews.text"].apply(is_english)]

# ==== Step 8: Save cleaned dataset ====
df.to_csv("cleaned_reviews.csv", index=False)

print(f"✅ Preprocessing complete! Saved {len(df)} cleaned reviews to 'cleaned_reviews.csv'")
