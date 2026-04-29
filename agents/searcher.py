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
        # Step 1: Generate 3 different search queries from the topic
        query_prompt = ChatPromptTemplate.from_template("""
You are a research expert. Generate 3 different search queries for the topic below.
Each query should focus on a different angle:
- Query 1: General overview
- Query 2: Latest news/developments
- Query 3: Data, statistics, or examples

Topic: {topic}

Return ONLY the 3 queries, one per line, no numbering or extra text.
""")
        query_chain = query_prompt | llm | StrOutputParser()
        queries_text = query_chain.invoke({"topic": topic})
        queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()][:3]

        print(f"🔍 Running {len(queries)} parallel queries:")
        for q in queries:
            print(f"   - {q}")

        # Step 2: Run all queries and collect results
        all_results = []
        for query in queries:
            try:
                results = search_tool.invoke(query)
                for r in results:
                    all_results.append(f"Query: {query}\nSource: {r['url']}\n{r['content']}")
            except Exception as e:
                print(f"⚠️ Query failed: {query} — {e}")

        # Step 3: Deduplicate results
        seen = set()
        unique_results = []
        for r in all_results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)

        combined = "\n\n---\n\n".join(unique_results)

        # Step 4: Summarize all findings
        summary_prompt = ChatPromptTemplate.from_template("""
You are a research assistant. Based on the following web search results from multiple queries,
provide a comprehensive and well-structured summary about: {topic}

Search Results:
{results}

Provide a detailed summary covering all angles found:
""")
        chain = summary_prompt | llm | StrOutputParser()
        summary = chain.invoke({"topic": topic, "results": combined})

        return summary

    print("✅ Multi-query Searcher agent ready")
    return search_and_summarize