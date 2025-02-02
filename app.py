import streamlit as st
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from modules.query_engine import query_ncert
from modules.question_recommend import recommend_questions

# Page configuration with favicon
st.set_page_config(
    page_title="NEET Exam AI Q&A",
    page_icon="assets/dna.png",
    layout="wide"
)

# Load environment variables and setup
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.error("❌ OpenAI API Key is missing. Set it in `.env` for local or `st.secrets` for deployment.")
    st.stop()

gpt4 = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize session state
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'recommended_questions' not in st.session_state:
    st.session_state.recommended_questions = []
if 'generated_answer' not in st.session_state:
    st.session_state.generated_answer = None

# Display Title with Image
col1, col2 = st.columns([1, 10])
with col1:
    st.image("assets/dna.png", width=50)  # ✅ Correctly loads image
with col2:
    st.title("NEET Exam AI Q&A")
st.subheader("Ask a question and get an AI-powered response!")


# Query Section
st.subheader("Ask your question")
user_query = st.text_input("🔍 Enter your question:", key="query_input")

# Add submit button
if st.button("Submit Question", key="submit_query"):
    if user_query:
        with st.spinner("🔎 Generating Answer..."):
            retrieved_context = query_ncert(user_query, top_k=3)
            combined_context = "\n".join(retrieved_context)
            
            gpt_prompt = [
                {"role": "system", "content": "You are an expert in NEET exam biology questions. Provide precise answers."},
                {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion:\n{user_query}\n\nAnswer:"}
            ]
            gpt_response = gpt4.invoke(gpt_prompt)
            st.session_state.generated_answer = gpt_response.content if hasattr(gpt_response, "content") else "⚠ Error: No response generated."
            
            st.session_state.recommended_questions = recommend_questions(user_query, top_k=5)

# Display the generated answer
if st.session_state.generated_answer:
    st.subheader("📝 AI Response")
    with st.container():
        st.write(st.session_state.generated_answer)

# Practice Questions Section
if st.session_state.recommended_questions:
    st.markdown("---")
    st.markdown("""
        <div class="title-container">
            <img src="assets/dna.png" width="30" class="dna-icon">
            <h2>Practice Questions</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with st.form(key="practice_questions"):
        all_questions_answered = True
        
        for idx, (question, options) in enumerate(st.session_state.recommended_questions, start=1):
            with st.container():
                st.write(f"\n### Question {idx}")
                st.write(question)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                valid_options = [opt for opt in options if opt.strip()]
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_answer = st.radio(
                        f"Select your answer:",
                        valid_options,
                        key=f"q{idx}",
                        index=None
                    )
                
                if selected_answer is None:
                    all_questions_answered = False
                
                if selected_answer:
                    st.session_state.user_answers[question] = selected_answer
                
                st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("Validate All Answers", use_container_width=True)
        
        if submit_button:
            if not all_questions_answered:
                st.warning("⚠️ Please answer all questions before validating.")
            else:
                st.markdown("""
                    <div class="title-container">
                        <img src="assets/dna.png" width="30" class="dna-icon">
                        <h2>Validation Results</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                for question, answer in st.session_state.user_answers.items():
                    validation_context = query_ncert(question, top_k=3)
                    combined_validation_context = "\n".join(validation_context)
                    
                    validation_prompt = [
                        {"role": "system", "content": "You are an expert in Biology subject of National eligibility cum entrance test (NEET) exam held in India for admission into undergraduate Medical courses. Verify if the given answer is correct based on the retrieved context."},
                        {"role": "user", "content": f"Context:\n{combined_validation_context}\n\nQuestion:\n{question}\n\nSelected Answer:\n{answer}\n\nIs this answer correct? Provide a brief explanation."}
                    ]
                    validation_response = gpt4.invoke(validation_prompt)
                    
                    with st.expander(f"Question {list(st.session_state.user_answers.keys()).index(question) + 1} Feedback"):
                        st.markdown(f"**Your Answer:** {answer}")
                        st.markdown("**Feedback:**")
                        st.write(validation_response.content if hasattr(validation_response, 'content') else '⚠ Error validating answer.')