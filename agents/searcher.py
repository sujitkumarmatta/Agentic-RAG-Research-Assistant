from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_searcher_agent():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    search_tool = TavilySearchResults(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5
    )

    def search_and_summarize(topic):

        # --- REACT STEP 1: REASON about what to search ---
        react_prompt = ChatPromptTemplate.from_template("""
You are a research agent using the ReAct framework.

THOUGHT: What do I need to find about this topic?
Analyze the topic and think about:
- What is the core question?
- What angles need to be covered?
- What specific data would be most useful?

ACTION PLAN: Generate exactly 3 search queries:
- Query 1: General overview angle
- Query 2: Latest news/developments angle  
- Query 3: Data, statistics, examples angle

Topic: {topic}

Return ONLY 3 queries, one per line, no numbering or labels.
""")
        query_chain = react_prompt | llm | StrOutputParser()
        queries_text = query_chain.invoke({"topic": topic})
        queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()][:3]

        print(f"\n🧠 ReAct THOUGHT: Analyzing topic: {topic}")
        print(f"📋 ReAct ACTION: Running {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"   Query {i}: {q}")

        # --- REACT STEP 2: ACT — run all queries ---
        all_results = []
        for query in queries:
            try:
                print(f"🔍 Searching: {query}")
                results = search_tool.invoke(query)
                for r in results:
                    all_results.append(
                        f"Query: {query}\nSource: {r['url']}\nContent: {r['content']}"
                    )
            except Exception as e:
                print(f"⚠️ Query failed: {query} — {e}")

        # --- REACT STEP 3: OBSERVE — review results ---
        seen = set()
        unique_results = []
        for r in all_results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)

        print(f"👁️ ReAct OBSERVATION: Got {len(unique_results)} unique results")

        combined = "\n\n---\n\n".join(unique_results)

        # --- REACT STEP 4: REASON again — summarize ---
        summary_prompt = ChatPromptTemplate.from_template("""
You are a research assistant completing a ReAct loop.

THOUGHT: I have gathered web search results. Now I need to:
1. Identify the most important findings
2. Check for contradictions between sources
3. Organize by relevance to the original topic
4. Synthesize into a coherent summary

OBSERVATION (search results):
{results}

ACTION: Write a comprehensive research summary about: {topic}

Cover:
- Main findings and key facts
- Latest developments
- Important statistics or data points
- Different perspectives if any

Provide detailed, well-organized findings:
""")
        chain = summary_prompt | llm | StrOutputParser()
        summary = chain.invoke({"topic": topic, "results": combined})

        print(f"✅ ReAct COMPLETE: Summary generated")
        return summary

    print("✅ ReAct Searcher agent ready")
    return search_and_summarize