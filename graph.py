from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.searcher import create_searcher_agent
from agents.writer import create_writer_agent, write_report
from dotenv import load_dotenv

load_dotenv()

# Shared memory that all agents read from and write to
class ResearchState(TypedDict):
    topic: str            # user's research question
    web_findings: str     # filled by searcher agent
    doc_findings: str     # filled by reader agent
    final_report: str     # filled by writer agent

def build_graph(reader_agent=None):
    # Initialize agents
    searcher = create_searcher_agent()
    writer_llm = create_writer_agent()

    # --- NODE 1: Web Search ---
    def search_web(state: ResearchState):
        print("🔎 Searcher agent working...")
        result = searcher(state["topic"])
        return {"web_findings": result}
        

    # --- NODE 2: Read Documents ---
    def read_documents(state: ResearchState):
        if reader_agent:
            print("📄 Reader agent working...")
            result = reader_agent.invoke(state["topic"])
            return {"doc_findings": result}
        else:
            print("⚠️ No document uploaded, skipping reader agent")
            return {"doc_findings": "No documents were uploaded by the user."}

    # --- NODE 3: Write Report ---
    def write_final_report(state: ResearchState):
        print("✍️ Writer agent working...")
        report = write_report(
            writer_llm,
            state["topic"],
            state["web_findings"],
            state["doc_findings"]
        )
        return {"final_report": report}

    # --- BUILD THE GRAPH ---
    graph = StateGraph(ResearchState)

    # Add all 3 nodes
    graph.add_node("search_web", search_web)
    graph.add_node("read_documents", read_documents)
    graph.add_node("write_report", write_final_report)

    # Define the flow: search → read → write → END
    graph.set_entry_point("search_web")
    graph.add_edge("search_web", "read_documents")
    graph.add_edge("read_documents", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()