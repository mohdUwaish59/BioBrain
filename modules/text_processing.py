import re
import logging
from langchain.text_splitter import TokenTextSplitter

# Setup logging
logging.basicConfig(filename="logs/text_processing.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextProcessor:
    def __init__(self, encoding="gpt2", chunk_size=512, chunk_overlap=100):
        """
        Initializes text processor with TokenTextSplitter.
        """
        self.text_splitter = TokenTextSplitter(
            encoding_name=encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logging.info(f"✅ Initialized TokenTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def clean_text(self, text):
        """
        Cleans and normalizes text by removing unwanted characters.
        """
        try:
            # Remove multiple newlines and extra spaces
            text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
            text = re.sub(r'\s+', ' ', text)  # Remove excessive spaces

            # Remove non-ASCII characters (optional)
            text = re.sub(r'[^\x00-\x7F]+', '', text)

            # Normalize dashes and hyphens
            text = text.replace("–", "-").replace("—", "-")

            logging.info("✅ Successfully cleaned text.")
            return text.strip()

        except Exception as e:
            logging.error(f"❌ Error cleaning text: {str(e)}", exc_info=True)
            return text  # Return original if error occurs

    def split_text(self, text):
        """
        Splits cleaned text into smaller token-based chunks.
        """
        try:
            cleaned_text = self.clean_text(text)
            chunks = self.text_splitter.split_text(cleaned_text)

            logging.info(f"✅ Successfully split text into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logging.error(f"❌ Failed to split text: {str(e)}", exc_info=True)
            return []
