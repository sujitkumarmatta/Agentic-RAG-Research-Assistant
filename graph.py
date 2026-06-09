from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.searcher import create_searcher_agent
from agents.writer import create_writer_agent, write_report
from fact_checker import create_fact_checker
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

class ResearchState(TypedDict):
    topic: str
    plan: str
    web_findings: str
    doc_findings: str
    final_report: str
    fact_check: str
    language: str
    has_documents: bool

def build_graph(reader_agent=None):
    searcher = create_searcher_agent()
    writer_llm = create_writer_agent()
    fact_checker = create_fact_checker()

    # --- NODE 0: PLANNER ---
    def create_plan(state: ResearchState):
        print("📋 Planner creating research plan...")

        # Skip detailed planning for simple short topics
        topic = state['topic']
        if len(topic.split()) <= 4:
            print("⚡ Simple topic — using quick plan")
            return {
                "plan": f"Quick research plan for: {topic}\n1. Search web\n2. Read docs if available\n3. Write report",
                "has_documents": reader_agent is not None
            }

        plan_prompt = f"""
You are a research planner. Create a clear step-by-step plan for: {topic}

Think about:
1. What web information is needed?
2. What document information is relevant?
3. What should the final report cover?
4. Key questions to answer?

Create a numbered research plan (5-7 steps):
"""
        plan = writer_llm.invoke(plan_prompt).content
        print("📋 Plan created")
        return {
            "plan": plan,
            "has_documents": reader_agent is not None
        }

    # --- ROUTER: decides path after planning ---
    def route_after_plan(state: ResearchState):
        print("🧭 Router making decision...")

        if state.get("has_documents", False):
            print("➡️  Route: Search + Documents (parallel)")
            return "search_and_read_parallel"
        else:
            print("➡️  Route: Web search only")
            return "search_only"

    # --- NODE 1A: SEARCH ONLY (no documents) ---
    def search_only(state: ResearchState):
        print("🌐 Running web search only...")

        enhanced_topic = f"""
Topic: {state['topic']}
Research Plan: {state.get('plan', '')}
Search for information that fulfills this plan.
"""
        web_findings = searcher(enhanced_topic)
        print("✅ Web search complete")

        return {
            "web_findings": web_findings,
            "doc_findings": "No documents were uploaded by the user."
        }

    # --- NODE 1B: PARALLEL SEARCH + READ (with documents) ---
    def search_and_read_parallel(state: ResearchState):
        print("🔀 Running Searcher + Reader in PARALLEL...")

        enhanced_topic = f"""
Topic: {state['topic']}
Research Plan: {state.get('plan', '')}
Search for information that fulfills this plan.
"""
        web_findings = "Search failed."
        doc_findings = "No documents were uploaded by the user."

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both at the same time
            web_future = executor.submit(searcher, enhanced_topic)
            doc_future = executor.submit(reader_agent, state["topic"]) if reader_agent else None

            # Collect results
            try:
                web_findings = web_future.result(timeout=60)
                print("✅ Web search complete")
            except Exception as e:
                print(f"⚠️ Web search failed: {e}")

            if doc_future:
                try:
                    doc_findings = doc_future.result(timeout=60)
                    print("✅ Document reading complete")
                except Exception as e:
                    print(f"⚠️ Document reading failed: {e}")

        return {
            "web_findings": web_findings,
            "doc_findings": doc_findings
        }

    # --- SOURCE EVALUATOR: checks web findings quality ---
    def evaluate_sources(state: ResearchState):
        print("🔍 Evaluating source quality...")

        eval_prompt = f"""
You are a source quality evaluator.

Evaluate these web research findings:
{state['web_findings'][:2000]}

Check:
1. Are claims supported by multiple sources?
2. Are there contradictions between sources?
3. Is the information recent and relevant?
4. Are there suspicious or unreliable claims?
5. Overall reliability score (0-10)

Keep evaluation brief — 3-5 bullet points max.
Flag any unreliable claims clearly.
"""
        evaluation = writer_llm.invoke(eval_prompt).content
        print("✅ Source evaluation complete")

        # Append evaluation to web findings so Writer is aware
        enhanced_findings = f"""
{state['web_findings']}

--- SOURCE QUALITY EVALUATION ---
{evaluation}
"""
        return {"web_findings": enhanced_findings}

    # --- NODE 2: WRITE REPORT ---
    def write_final_report(state: ResearchState):
        print("✍️ Writer agent working...")
        report = write_report(
            writer_llm,
            state["topic"],
            state["web_findings"],
            state["doc_findings"],
            state.get("language", "English"),
            state.get("plan", "")
        )
        return {"final_report": report}

    # --- QUALITY ROUTER: checks if report needs rewriting ---
    def route_after_writing(state: ResearchState):
        print("🧭 Checking report quality...")

        quality_prompt = f"""
Rate this research report quality on ONE criterion only:
Is it comprehensive and well-structured?

Report (first 500 chars): {state['final_report'][:500]}

Reply with ONLY: GOOD or NEEDS_WORK
"""
        result = writer_llm.invoke(quality_prompt).content.strip()
        print(f"📊 Report quality: {result}")

        if "NEEDS_WORK" in result:
            return "rewrite"
        return "verify"

    # --- NODE 2B: REWRITE REPORT (if quality check fails) ---
    def rewrite_report(state: ResearchState):
        print("🔄 Rewriting report — quality was insufficient...")

        rewrite_prompt = f"""
The following research report needs improvement.
Make it more comprehensive, better structured and clearer.

Original Report:
{state['final_report']}

Topic: {state['topic']}
Web Findings: {state['web_findings'][:1000]}
Doc Findings: {state['doc_findings'][:1000]}

Write an improved version in {state.get('language', 'English')}:
"""
        improved = writer_llm.invoke(rewrite_prompt).content
        print("✅ Report rewritten")
        return {"final_report": improved}

    # --- NODE 3: FACT CHECK ---
    def verify_facts(state: ResearchState):
        print("🛡️ Fact checker running...")
        verification = fact_checker(
            state["final_report"],
            state["web_findings"],
            state["doc_findings"]
        )
        return {"fact_check": verification}

    # ─────────────────────────────────────────
    # BUILD THE GRAPH
    # ─────────────────────────────────────────
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("create_plan", create_plan)
    graph.add_node("search_only", search_only)
    graph.add_node("search_and_read_parallel", search_and_read_parallel)
    graph.add_node("evaluate_sources", evaluate_sources)
    graph.add_node("write_report", write_final_report)
    graph.add_node("rewrite_report", rewrite_report)
    graph.add_node("verify_facts", verify_facts)

    # Entry point
    graph.set_entry_point("create_plan")

    # Router 1: after planning — decides search strategy
    graph.add_conditional_edges(
        "create_plan",
        route_after_plan,
        {
            "search_only": "search_only",
            "search_and_read_parallel": "search_and_read_parallel"
        }
    )

    # Both search paths go to source evaluator
    graph.add_edge("search_only", "evaluate_sources")
    graph.add_edge("search_and_read_parallel", "evaluate_sources")

    # Source evaluator goes to writer
    graph.add_edge("evaluate_sources", "write_report")

    # Router 2: after writing — quality check
    graph.add_conditional_edges(
        "write_report",
        route_after_writing,
        {
            "verify": "verify_facts",
            "rewrite": "rewrite_report"
        }
    )

    # Rewrite goes to fact check
    graph.add_edge("rewrite_report", "verify_facts")

    # Fact check ends
    graph.add_edge("verify_facts", END)

    return graph.compile()