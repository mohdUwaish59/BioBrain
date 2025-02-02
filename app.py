import streamlit as st
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # ✅ Correct class for GPT-4
from modules.query_engine import query_ncert
from modules.question_recommend import recommend_questions

# Load environment variables from .env (for local testing)
load_dotenv()

# Use Streamlit secrets for deployment, fallback to .env for local testing
#openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Check if API key exists
if not openai_api_key:
    st.error("❌ OpenAI API Key is missing. Set it in `.env` for local or `st.secrets` for deployment.")
    st.stop()

# ✅ Use ChatOpenAI instead of OpenAI
gpt4 = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

# Setup logging
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit UI
st.title("📚 NEET Exam AI Q&A")
st.subheader("Ask a question and get an AI-powered response!")

# User input
user_query = st.text_input("🔍 Enter your question:")

if user_query:
    with st.spinner("🔎 Generating Answer..."):
        retrieved_context = query_ncert(user_query, top_k=3)
        combined_context = "\n".join(retrieved_context)

        # Generate Answer using GPT-4
        gpt_prompt = [
            {"role": "system", "content": "You are an expert in NEET exam biology questions. Provide precise answers."},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion:\n{user_query}\n\nAnswer:"}
        ]
        gpt_response = gpt4.invoke(gpt_prompt)  # ✅ Fixes OpenAI API error

        # ✅ Extract only the "content" field correctly
        generated_answer = gpt_response.content if hasattr(gpt_response, "content") else "⚠ Error: No response generated."

        # Recommend Related Questions
        recommended_questions = recommend_questions(user_query, top_k=5)

    # ✅ Print only the AI-generated response
    st.subheader("📝 AI Response")
    st.write(generated_answer)

    # ✅ Display recommended questions with answer selection
    st.subheader("🔁 Related Questions (Select Your Answer)")

    user_answers = {}  # Dictionary to store user-selected answers

    for idx, (question, options) in enumerate(recommended_questions, start=1):
        st.write(f"\n🔹 {idx}. {question}")

        # Prepare radio button choices
        valid_options = [opt for opt in options if opt.strip()]  # Only show non-empty options
        selected_answer = st.radio(f"Select your answer for Question {idx}:", valid_options, key=f"q{idx}")

        # Store selected answer
        user_answers[question] = selected_answer

    # ✅ Validate Answers using RAG Approach
    if st.button("🔍 Validate Answers"):
        st.subheader("✅ Answer Validation Results")

        for question, answer in user_answers.items():
            # Retrieve relevant context for the selected answer
            validation_context = query_ncert(question, top_k=3)
            combined_validation_context = "\n".join(validation_context)

            # Ask GPT-4 if the selected answer is correct
            validation_prompt = [
                {"role": "system", "content": "You are an expert in NEET exam biology. Verify if the given answer is correct based on the retrieved context."},
                {"role": "user", "content": f"Context:\n{combined_validation_context}\n\nQuestion:\n{question}\n\nSelected Answer:\n{answer}\n\nIs this answer correct? Provide a brief explanation."}
            ]
            validation_response = gpt4.invoke(validation_prompt)

            # ✅ Display validation result
            st.write(f"\n🔹 **Question:** {question}")
            st.write(f"   ✅ **Your Answer:** {answer}")
            st.write(f"   📖 **Validation Result:** {validation_response.content if hasattr(validation_response, 'content') else '⚠ Error validating answer.'}")
