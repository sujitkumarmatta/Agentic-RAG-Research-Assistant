from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_fact_checker():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    def verify_report(report, web_findings, doc_findings):
        print("🛡️ Fact checker verifying report...")

        verify_prompt = ChatPromptTemplate.from_template("""
You are a rigorous fact-checker. Verify this research report.

REPORT TO VERIFY:
{report}

SOURCES AVAILABLE:
Web Research: {web_findings}
Document Research: {doc_findings}

For each major claim in the report:
1. Find the source that supports it
2. Mark as VERIFIED ✅ or UNVERIFIED ❌ or PARTIALLY VERIFIED ⚠️

Then provide:
VERIFICATION SUMMARY:
- Total claims checked: [N]
- Verified: [N]
- Unverified: [N]
- Partially verified: [N]
- Overall reliability: [High/Medium/Low]

UNVERIFIED CLAIMS (if any):
[List claims that could not be verified]

RECOMMENDATION:
[Should report be used as-is, with caution, or needs revision?]
""")
        chain = verify_prompt | llm | StrOutputParser()
        verification = chain.invoke({
            "report": report,
            "web_findings": web_findings,
            "doc_findings": doc_findings
        })

        print("✅ Fact checking complete")
        return verification

    return verify_report