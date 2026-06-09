import trafilatura
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_from_url(url):
    print(f"🔗 Loading content from: {url}")

    # Download and extract clean text from URL
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print("⚠️ Could not fetch URL")
        return []

    text = trafilatura.extract(downloaded)
    if not text:
        print("⚠️ Could not extract text from URL")
        return []

    # Create a document with URL as source
    doc = Document(
        page_content=text,
        metadata={"source_file": url, "page": 0}
    )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents([doc])
    print(f"✅ URL loaded: {len(chunks)} chunks from {url}")
    return chunks