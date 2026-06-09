from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import os
import numpy as np

def build_raptor_tree(chunks, embeddings_model):
    """
    Builds a 3-level tree:
    Level 0: Original chunks (leaf nodes)
    Level 1: Section summaries (groups of chunks)
    Level 2: Document summary (top level)
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    summarize_prompt = ChatPromptTemplate.from_template("""
Summarize the following text concisely but completely.
Preserve all important facts, numbers, dates and names.

Text:
{text}

Summary:
""")
    chain = summarize_prompt | llm | StrOutputParser()

    print(f"🌲 Building RAPTOR tree from {len(chunks)} chunks...")

    # Level 0: Original chunks
    all_docs = list(chunks)

    # Level 1: Group chunks into sections of 5 and summarize
    section_summaries = []
    group_size = 5
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        group_text = "\n\n".join([c.page_content for c in group])

        try:
            summary = chain.invoke({"text": group_text[:3000]})
            source = group[0].metadata.get("source_file", "unknown")
            section_doc = Document(
                page_content=f"[SECTION SUMMARY] {summary}",
                metadata={
                    "source_file": source,
                    "level": "section",
                    "page": f"sections_{i}-{i+group_size}"
                }
            )
            section_summaries.append(section_doc)
            print(f"✅ Section {i//group_size + 1} summarized")
        except Exception as e:
            print(f"⚠️ Section summary failed: {e}")

    all_docs.extend(section_summaries)

    # Level 2: Full document summary
    if section_summaries:
        all_sections = "\n\n".join([s.page_content for s in section_summaries])
        try:
            doc_summary = chain.invoke({"text": all_sections[:4000]})
            doc_doc = Document(
                page_content=f"[DOCUMENT SUMMARY] {doc_summary}",
                metadata={
                    "source_file": "full_document",
                    "level": "document",
                    "page": "full"
                }
            )
            all_docs.append(doc_doc)
            print("✅ Full document summary created")
        except Exception as e:
            print(f"⚠️ Document summary failed: {e}")

    print(f"🌲 RAPTOR tree built: {len(chunks)} chunks + {len(section_summaries)} section summaries + 1 doc summary")
    return all_docs