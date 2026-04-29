from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_and_split_pdf(file_path):
    # Step 1: Open and read the PDF file
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Filter out empty pages
    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        print("⚠️ No text found in this PDF - it may be a scanned/image PDF")
        return []

    # Step 3: Split the text into small chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"✅ PDF loaded: {len(chunks)} chunks created")
    return chunks