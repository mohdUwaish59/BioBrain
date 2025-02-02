import logging
import faiss
import os
import pickle
import numpy as np
from langchain.evaluation.qa import QAEvaluator
from langchain.evaluation.embedding_distance import EmbeddingDistanceEvaluator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(filename="logs/lc_evaluation.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_QA_INDEX_PATH = "faiss_qa"

class LangChainEvaluator:
    def __init__(self):
        """
        Initializes LangChain's evaluation modules.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.qa_evaluator = QAEvaluator()
        self.embedding_evaluator = EmbeddingDistanceEvaluator()

        logging.info("✅ Initialized LangChain Evaluators.")

    def precision_at_k(self, recommended, relevant, k=5):
        """
        Computes Precision @ K using LangChain's QAEvaluator.
        """
        recommended = recommended[:k]
        precision_score = self.qa_evaluator.evaluate([{"question": q, "answer": q in relevant} for q in recommended])

        precision = precision_score["score"]
        logging.info(f"✅ Precision @ {k}: {precision:.4f}")
        return precision

    def recall_at_k(self, recommended, relevant, k=5):
        """
        Computes Recall @ K using LangChain.
        """
        recommended = recommended[:k]
        recall_score = sum(1 for q in recommended if q in relevant) / len(relevant) if relevant else 0

        logging.info(f"✅ Recall @ {k}: {recall_score:.4f}")
        return recall_score

    def embedding_similarity(self, recommended, relevant):
        """
        Computes similarity between recommended and relevant questions.
        """
        eval_pairs = [{"reference": r, "prediction": recommended[0]} for r in relevant if recommended]
        similarity_scores = self.embedding_evaluator.evaluate(eval_pairs)

        avg_similarity = np.mean([s["score"] for s in similarity_scores]) if similarity_scores else 0
        logging.info(f"✅ Embedding Similarity: {avg_similarity:.4f}")
        return avg_similarity
