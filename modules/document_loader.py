import os
import logging
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PDFMinerLoader, PyMuPDFLoader

# Setup logging
logging.basicConfig(filename="logs/document_loader.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LOADER_TYPES = {
    "pypdf": PyPDFLoader,
    "unstructured": UnstructuredPDFLoader,
    "pdfminer": PDFMinerLoader,
    "pymupdf": PyMuPDFLoader
}

def load_pdfs(pdf_folder: str, loader_type: str = "pypdf"):
    """Loads PDFs from a folder using LangChain document loaders."""
    all_docs = []
    LoaderClass = LOADER_TYPES.get(loader_type, PyPDFLoader)

    try:
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, file)
                logging.info(f"Loading: {pdf_path} using {loader_type}")
                loader = LoaderClass(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
        logging.info(f"Successfully loaded {len(all_docs)} documents from {pdf_folder}")
    except Exception as e:
        logging.error(f"Error loading PDFs: {str(e)}", exc_info=True)

    return all_docs
