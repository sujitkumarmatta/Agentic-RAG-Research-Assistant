from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_adaptive_retriever(base_retriever):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    def adaptive_retrieve(question, max_attempts=3):
        print(f"🔁 Adaptive retrieval starting for: {question}")

        all_docs = []
        current_query = question
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print(f"🔍 Attempt {attempt}: searching with query: {current_query}")

            # Retrieve docs
            docs = base_retriever.invoke(current_query)
            all_docs.extend(docs)

            # Deduplicate
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
            all_docs = unique_docs

            # Evaluate if results are sufficient
            context = "\n\n".join([d.page_content for d in all_docs[:5]])

            eval_prompt = ChatPromptTemplate.from_template("""
Evaluate if the retrieved context sufficiently answers the question.

Question: {question}
Retrieved Context: {context}

Answer with ONLY one of:
SUFFICIENT - context fully answers the question
INSUFFICIENT - need more specific information
IRRELEVANT - context is completely off-topic

Then on next line explain why in one sentence.
""")
            eval_chain = eval_prompt | llm | StrOutputParser()
            evaluation = eval_chain.invoke({
                "question": question,
                "context": context
            })

            print(f"📊 Retrieval evaluation: {evaluation[:50]}...")

            if "SUFFICIENT" in evaluation:
                print(f"✅ Sufficient context found after {attempt} attempt(s)")
                break

            if attempt < max_attempts:
                # Generate better query
                requery_prompt = ChatPromptTemplate.from_template("""
The previous search was insufficient. Generate a better search query.

Original question: {question}
Previous query: {current_query}
Why insufficient: {evaluation}

Generate ONE improved, more specific search query.
Return ONLY the query, nothing else.
""")
                requery_chain = requery_prompt | llm | StrOutputParser()
                current_query = requery_chain.invoke({
                    "question": question,
                    "current_query": current_query,
                    "evaluation": evaluation
                }).strip()
                print(f"🔄 Retrying with better query: {current_query}")

        print(f"✅ Adaptive retrieval complete: {len(all_docs)} total docs")
        return all_docs[:10]

    return adaptive_retrieve