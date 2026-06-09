from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_writer_agent():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    print("✅ Writer agent ready")
    return llm

def write_report(llm, topic, web_findings, doc_findings, language="English", plan=""):
    prompt = f"""
You are a professional research report writer using step-by-step planning.

STEP 1 — REVIEW THE PLAN:
{plan if plan else "Use standard research structure"}

STEP 2 — REVIEW FINDINGS:
Web Research: {web_findings}
Document Research: {doc_findings}

STEP 3 — CHECK COMPLETENESS:
Does the findings cover all points in the plan?
Note any gaps or missing information.

STEP 4 — WRITE THE REPORT:
Research Topic: {topic}

Write a detailed professional report with:
1. Executive Summary
2. Research Plan Overview
3. Key Findings from Web Research
4. Key Findings from Documents
5. Analysis & Insights
6. Conclusion & Recommendations

IMPORTANT: Write entire report in {language} language only.
"""
    response = llm.invoke(prompt)
    return response.content

def write_summary(llm, content, language="English"):
    prompt = f"""
Summarize the following content clearly and concisely.

Cover:
1. 📌 Main Topic
2. 🔑 Key Points (bullet points)
3. 📊 Important Data/Numbers
4. 📝 Brief Summary (2-3 sentences)

Content:
{content}

IMPORTANT: Respond in {language} language only.
"""
    response = llm.invoke(prompt)
    return response.content

def write_comparison(llm, topic, doc1_name, doc1_content, doc2_name, doc2_content, language="English"):
    prompt = f"""
You are a document comparison expert.
Compare the two documents on topic: {topic}

Document 1 ({doc1_name}):
{doc1_content}

Document 2 ({doc2_name}):
{doc2_content}

Provide:
1. 📄 Document 1 Summary
2. 📄 Document 2 Summary
3. ✅ Similarities
4. ❌ Differences
5. 📊 Key Takeaway

IMPORTANT: Respond in {language} language only.
"""
    response = llm.invoke(prompt)
    return response.content

def write_key_points(llm, content, language="English"):
    prompt = f"""
Extract the most important key points from this content.

Format as:
🔑 KEY POINTS:
- Point 1
- Point 2
- Point 3
...

Then add:
💡 MAIN INSIGHT: (one sentence)

Content:
{content}

IMPORTANT: Respond in {language} language only.
"""
    response = llm.invoke(prompt)
    return response.content

def write_email_body(llm, report, topic, language="English"):
    prompt = f"""
Convert this research report into a professional email body.

Keep it:
- Concise (max 300 words)
- Professional tone
- Highlight top 3 findings
- End with clear next steps

Report:
{report}

Topic: {topic}

IMPORTANT: Respond in {language} language only.
"""
    response = llm.invoke(prompt)
    return response.content