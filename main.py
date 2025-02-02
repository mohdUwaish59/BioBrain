import logging
from langchain.schema import Document  # Import Document for proper formatting
from modules.document_loader import load_pdfs
from modules.text_processing import TextProcessor
from modules.embeddings_store import store_in_faiss
from modules.query_engine import query_ncert

# Setup logging
logging.basicConfig(filename="logs/main.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
PDF_FOLDER = "dataset/data"

try:
    # Step 1️⃣: Load NCERT PDFs
    docs = load_pdfs(PDF_FOLDER, loader_type="pypdf")
    #
    if not docs:
        raise ValueError("No documents were loaded. Check the PDF folder path.")
#
    ## Step 2️⃣: Initialize Text Processor
    text_processor = TextProcessor(chunk_size=512, chunk_overlap=100)
#
    ## Step 3️⃣: Process Documents (Clean & Token-Based Chunking)
    split_texts = [chunk for doc in docs for chunk in text_processor.split_text(doc.page_content)]
#
    ## Step 4️⃣: Convert Split Texts into `Document` Objects for FAISS
    split_docs = [Document(page_content=text) for text in split_texts]
#
    ## Step 5️⃣: Store in FAISS
    store_in_faiss(split_docs)

    # Step 6️⃣: Query Retrieval Test
    test_query = """
    Which of the following groups is NOT included in the Kingdom Plantae?
    """
    results = query_ncert(test_query)

    # Step 7️⃣: Display Retrieved Results
    print("\n🔎 Search Results for Query:", test_query)
    for i, text in enumerate(results):
        print(f"\n🔹 Result {i+1}:\n{text}")

except Exception as e:
    logging.error(f"❌ Main execution error: {str(e)}", exc_info=True)
