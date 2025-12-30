import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_reviews.csv")

# Step 1: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate

# Step 2: Create embeddings
embeddings = model.encode(df["reviews.text"].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Step 3: Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 4: Save FAISS index
faiss.write_index(index, "reviews.index")

# Save metadata for later retrieval
df.to_csv("reviews_metadata.csv", index=False)

print(f"✅ Stored {len(embeddings)} review embeddings in FAISS index.")
