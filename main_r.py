import logging
from modules.question_store import store_questions
from modules.question_recommend import recommend_questions

# Setup logging
logging.basicConfig(filename="logs/main.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CSV_PATH = "dataset/questions.csv"  # Ensure this is the correct path to your dataset

try:
    # Step 1️⃣: Store Question Embeddings
    #store_questions(CSV_PATH)

    # Step 2️⃣: Recommend Questions
    test_query = "Which of the following groups is NOT included in the Kingdom Plantae?"
    recommendations = recommend_questions(test_query)

    # Step 3️⃣: Display Recommended Questions
    print("\n🔎 Recommended Questions for Query:", test_query)
    for i, question in enumerate(recommendations):
        print(f"\n🔹 Recommended {i+1}: {question}")

except Exception as e:
    logging.error(f"❌ Main execution error: {str(e)}", exc_info=True)
