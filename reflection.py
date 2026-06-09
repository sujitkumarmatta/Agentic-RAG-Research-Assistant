from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_reflection_agent():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    def reflect_and_improve(question, answer, context):
        # Step 1: Critique the answer
        critique_prompt = ChatPromptTemplate.from_template("""
You are a critical evaluator. Analyze this answer:

Question: {question}
Answer: {answer}
Context used: {context}

Evaluate on:
1. ACCURACY — Is everything factually correct based on context?
2. COMPLETENESS — Does it fully answer the question?
3. CLARITY — Is it clear and well structured?
4. HALLUCINATION — Any claims not supported by context?

Give specific critique points. Be harsh but fair.
Format:
ACCURACY: [score 1-10] [issues if any]
COMPLETENESS: [score 1-10] [what's missing]
CLARITY: [score 1-10] [issues if any]
HALLUCINATION RISK: [Low/Medium/High] [suspicious claims]
OVERALL: [score 1-10]
NEEDS IMPROVEMENT: [Yes/No]
""")
        critique_chain = critique_prompt | llm | StrOutputParser()
        critique = critique_chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })

        print(f"🔄 Self-reflection critique:\n{critique}")

        # Step 2: Check if improvement needed
        needs_improvement = "NEEDS IMPROVEMENT: Yes" in critique or \
                           "HALLUCINATION RISK: High" in critique or \
                           "HALLUCINATION RISK: Medium" in critique

        if not needs_improvement:
            print("✅ Answer passed self-reflection — no improvement needed")
            return answer, critique

        # Step 3: Improve the answer
        improve_prompt = ChatPromptTemplate.from_template("""
You are improving an answer based on critique.

Original Question: {question}
Original Answer: {answer}
Critique: {critique}
Context: {context}

Based on the critique, write an improved answer that:
1. Fixes all accuracy issues
2. Adds any missing information from context
3. Removes any hallucinated claims
4. Is clearer and better structured

Only use information from the context.
Improved Answer:
""")
        improve_chain = improve_prompt | llm | StrOutputParser()
        improved = improve_chain.invoke({
            "question": question,
            "answer": answer,
            "critique": critique,
            "context": context
        })

        print("✅ Answer improved through self-reflection")
        return improved, critique

    return reflect_and_improve