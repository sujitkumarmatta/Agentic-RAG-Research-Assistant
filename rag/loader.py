from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def normalize_numbers(text):
    """Convert Indian number format to plain numbers"""
    # Convert 8,25,517 → 825517
    def replace_indian(match):
        return match.group().replace(",", "")

    # Indian lakh format: X,XX,XXX
    text = re.sub(r'\d{1,2},\d{2},\d{3}', replace_indian, text)
    # Indian crore format: X,XX,XX,XXX
    text = re.sub(r'\d{1,2},\d{2},\d{2},\d{3}', replace_indian, text)
    # Western thousand format: XX,XXX or XXX,XXX
    text = re.sub(r'\d{1,3},\d{3}(?!,\d)', replace_indian, text)
    return text

def load_and_split_pdf(file_path, file_name=None):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Filter empty pages
    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        print(f"⚠️ No text found in {file_name}")
        return []

    # Add filename to metadata + normalize numbers
    for doc in documents:
        doc.metadata["source_file"] = file_name or file_path
        doc.page_content = normalize_numbers(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ {file_name}: {len(chunks)} chunks created")
    return chunks