import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load metadata and FAISS index
df = pd.read_csv("reviews_metadata.csv")
index = faiss.read_index("reviews.index")

# Load same embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_reviews(query, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Return top reviews
    results = []
    for idx in indices[0]:
        results.append(df.iloc[idx]["reviews.text"])
    return results

# Example usage
query = "What do customers say about battery life?"
top_reviews = search_reviews(query)

print("🔍 Top Reviews Found:")
for r in top_reviews:
    print("-", r)
