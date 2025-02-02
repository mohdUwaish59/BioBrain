import logging
import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(filename="logs/embeddings_store.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_INDEX_PATH = "faiss_index"

def store_in_faiss(docs):
    """
    Stores document embeddings in FAISS.
    """
    try:
        # Step 1️⃣: Load Embedding Model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Step 2️⃣: Extract Texts for Embeddings
        texts = [doc.page_content for doc in docs]  # ✅ Now correctly formatted

        # Step 3️⃣: Generate Embeddings
        embeddings = model.encode(texts).astype(np.float32)

        # Step 4️⃣: Initialize FAISS Index
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embeddings)  # ✅ Store embeddings

        # Step 5️⃣: Save FAISS Index
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(FAISS_INDEX_PATH, "faiss_index.bin"))

        # Step 6️⃣: Save Text Chunks
        with open(os.path.join(FAISS_INDEX_PATH, "texts.pkl"), "wb") as f:
            pickle.dump(texts, f)

        logging.info(f"✅ Successfully stored {len(texts)} document chunks in FAISS.")

    except Exception as e:
        logging.error(f"❌ Error storing embeddings in FAISS: {str(e)}", exc_info=True)
