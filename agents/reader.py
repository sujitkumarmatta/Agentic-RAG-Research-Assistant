from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

def create_reader_agent(retriever):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"

    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below.

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ Reader agent ready")
    return qa_chain