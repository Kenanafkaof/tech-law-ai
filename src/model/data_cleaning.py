from sentence_transformers import SentenceTransformer
import pandas as pd

import faiss
import numpy as np

# Load the data
df = pd.read_csv('../datasets/all_tech_cases_5year.csv')

df = df.drop_duplicates().reset_index(drop=True)

# Combine relevant text fields into a single field for embedding
df['combined_text'] = df['case_name'].fillna('') + " " + df['text_excerpt'].fillna('')

# Load a pre-trained sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for each case
documents = df['combined_text'].tolist()
embeddings = embedder.encode(documents, show_progress_bar=True)


# Determine the dimensionality of the embeddings
embedding_dim = embeddings.shape[1]

# Create a FAISS index (using L2 distance; you can also explore cosine similarity with some adjustments)
nlist = 10  # number of clusters (tune this)
quantizer = faiss.IndexFlatL2(embedding_dim)
index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)


# Add the document embeddings to the index
embeddings_np = np.array(embeddings, dtype=np.float32)
index.add(embeddings_np)

index.train(embeddings_np)
index.add(embeddings_np)

print("Number of documents in index:", index.ntotal)

# Define a query – this could be any legal scenario input by a user
query = "A dispute involving intellectual property and trademark issues in the technology sector."

# Compute the query embedding
query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)

# Number of nearest neighbors to retrieve
k = 5

# Search the index
distances, indices = index.search(query_embedding, k)

# Display the results
print("Top k similar cases:")
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}:")
    print("Case Name:", df.iloc[idx]['case_name'])
    print("Excerpt:", df.iloc[idx]['text_excerpt'][:300], "...\n")
    