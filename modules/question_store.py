import logging
import faiss
import numpy as np
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(filename="logs/question_store.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_QA_INDEX_PATH = "faiss_qa"

def store_questions(csv_path):
    """
    Stores question embeddings from a CSV dataset in FAISS for recommendation.
    """
    try:
        # Load dataset
        df = pd.read_csv(csv_path)

        # Ensure 'Question' column exists
        if "Question" not in df.columns:
            raise ValueError("Dataset must have a 'Question' column.")

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Extract questions
        questions = df["Question"].dropna().tolist()
        embeddings = model.encode(questions).astype(np.float32)

        # Initialize FAISS Index
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embeddings)

        # Save FAISS index
        os.makedirs(FAISS_QA_INDEX_PATH, exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(FAISS_QA_INDEX_PATH, "qa_index.bin"))

        # Save question text data
        with open(os.path.join(FAISS_QA_INDEX_PATH, "questions.pkl"), "wb") as f:
            pickle.dump(questions, f)

        logging.info(f"✅ Successfully stored {len(questions)} question embeddings in FAISS.")

    except Exception as e:
        logging.error(f"❌ Error storing questions in FAISS: {str(e)}", exc_info=True)
