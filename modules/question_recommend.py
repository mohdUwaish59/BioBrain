import logging
import faiss
import numpy as np
import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(filename="logs/question_recommend.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_QA_INDEX_PATH = "faiss_qa"
CSV_PATH = "dataset/questions.csv"  # Ensure this is the correct path

def recommend_questions(query_text, top_k=5):
    """
    Retrieves similar questions along with their answer options.
    """
    try:
        # Load the dataset to get questions and options
        df = pd.read_csv(CSV_PATH)

        # Ensure the dataset has necessary columns
        if "Question" not in df.columns:
            raise ValueError("Dataset must have a 'Question' column.")

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Extract questions and options
        questions = df["Question"].dropna().tolist()
        option_columns = [col for col in df.columns if col not in ["Subject", "Question"]]
        options = df[option_columns].fillna("").astype(str).values.tolist()

        # Load FAISS index
        faiss_index_path = os.path.join(FAISS_QA_INDEX_PATH, "qa_index.bin")
        if not os.path.exists(faiss_index_path):
            logging.error("❌ FAISS QA index not found.")
            return []

        faiss_index = faiss.read_index(faiss_index_path)

        # Load stored questions
        questions_path = os.path.join(FAISS_QA_INDEX_PATH, "questions.pkl")
        with open(questions_path, "rb") as f:
            stored_questions = pickle.load(f)

        # Encode query
        query_embedding = model.encode([query_text]).astype(np.float32)
        _, indices = faiss_index.search(query_embedding, top_k)

        # Retrieve recommended questions along with options
        recommended_questions = []
        for i in indices[0]:
            if i < len(stored_questions):
                question_text = stored_questions[i]
                question_options = options[i] if i < len(options) else ["", "", "", "", "", ""]
                recommended_questions.append((question_text, question_options))

        logging.info(f"✅ Recommended {len(recommended_questions)} questions for query: {query_text}")
        return recommended_questions

    except Exception as e:
        logging.error(f"❌ Error recommending questions: {str(e)}", exc_info=True)
        return []
