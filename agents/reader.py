from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_reader_agent(retriever):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you can't find the answer, say "I couldn't find relevant info in the document."

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def run(query):
        # Manually invoke retriever instead of using pipe operator
        docs = retriever.invoke(query)
        context = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    print("✅ Reader agent ready")
    return run