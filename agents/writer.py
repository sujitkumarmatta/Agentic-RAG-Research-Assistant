from langchain_groq import ChatGroq
import os

def create_writer_agent():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"    )
    print("✅ Writer agent ready")
    return llm

def write_report(llm, topic, web_findings, doc_findings):
    prompt = f"""
    You are a professional research report writer.

    Research Topic: {topic}

    === WEB RESEARCH FINDINGS ===
    {web_findings}

    === DOCUMENT RESEARCH FINDINGS ===
    {doc_findings}

    Based on the above findings, write a detailed and professional research report with these sections:

    1. Executive Summary
    2. Key Findings from Web Research
    3. Key Findings from Documents
    4. Analysis & Insights
    5. Conclusion

    Make the report clear, structured, and professional.
    """

    response = llm.invoke(prompt)
    return response.content