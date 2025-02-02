import logging
import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(filename="logs/query_engine.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_INDEX_PATH = "faiss_index"

def query_ncert(query_text, top_k=3):
    """
    Retrieves relevant NCERT content from FAISS using GPT-2 embeddings.
    """
    try:
        # Step 1️⃣: Load Embedding Model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Step 2️⃣: Load FAISS Index
        faiss_index_path = os.path.join(FAISS_INDEX_PATH, "faiss_index.bin")
        if not os.path.exists(faiss_index_path):
            logging.error("❌ FAISS index not found.")
            return []

        faiss_index = faiss.read_index(faiss_index_path)

        # Step 3️⃣: Load Stored Text Chunks
        texts_path = os.path.join(FAISS_INDEX_PATH, "texts.pkl")
        with open(texts_path, "rb") as f:
            stored_texts = pickle.load(f)

        # Step 4️⃣: Encode Query Text
        query_embedding = model.encode([query_text]).astype(np.float32)

        # Step 5️⃣: Search FAISS for Nearest Matches
        _, indices = faiss_index.search(query_embedding, top_k)

        # Step 6️⃣: Retrieve Relevant Text Chunks
        retrieved_texts = [stored_texts[i] for i in indices[0] if i < len(stored_texts)]

        logging.info(f"✅ Query: {query_text} | Retrieved {len(retrieved_texts)} results")
        return retrieved_texts

    except Exception as e:
        logging.error(f"❌ Error querying FAISS: {str(e)}", exc_info=True)
        return []
