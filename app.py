import streamlit as st
from dotenv import load_dotenv
from graph import build_graph
from rag.loader import load_and_split_pdf
from rag.retriever import create_vector_store, get_retriever
from agents.reader import create_reader_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os
from fpdf import FPDF

def export_chat_as_pdf(chat_history, title="Chat Export"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(5)

    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"{role}:", ln=True)
        pdf.set_font("Helvetica", "", 10)

        # Handle special characters
        content = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, content)
        pdf.ln(3)

    return pdf.output()

load_dotenv()

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Multi-Agent Research Assistant")
st.markdown("*Powered by LangGraph + Advanced RAG + Groq*")
st.markdown("---")

# --- MODE SELECTOR ---
mode = st.radio(
    "Choose Mode:",
    ["📋 Research Report Generation", "💬 PDF Q&A Chat", "🔄 Compare PDFs"],
    horizontal=True
)

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Upload PDFs (Optional)")
    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Process PDFs", use_container_width=True):
            all_chunks = []
            pdf_names = []

            with st.spinner("Reading and indexing..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    chunks = load_and_split_pdf(tmp_path, uploaded_file.name)
                    os.unlink(tmp_path)

                    if chunks:
                        all_chunks.extend(chunks)
                        pdf_names.append(uploaded_file.name)
                        st.sidebar.write(f"✅ {uploaded_file.name} — {len(chunks)} chunks")
                    else:
                        st.sidebar.write(f"⚠️ {uploaded_file.name} — no text found, skipped")

            if not all_chunks:
                st.error("⚠️ Could not extract text from any PDF. Make sure they are not scanned/image based.")
            else:
                vector_store, stored_chunks = create_vector_store(all_chunks)
                retriever = get_retriever(vector_store, stored_chunks)
                st.session_state.retriever = retriever
                st.session_state.pdf_names = pdf_names
                st.session_state.pdf_chat_history = []
                st.session_state.total_chunks = len(all_chunks)
                st.session_state.doc_summary = None
                st.success(f"✅ {len(pdf_names)} PDF(s) indexed! ({len(all_chunks)} total chunks)")

    if "retriever" in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Loaded PDFs")
        for name in st.session_state.pdf_names:
            st.markdown(f"- {name}")
        st.markdown(f"**Total chunks:** {st.session_state.total_chunks}")

        st.markdown("---")
        if st.button("📝 Summarize Documents", use_container_width=True):
            with st.spinner("Summarizing..."):
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile"
                )

                sample_docs = st.session_state.retriever.invoke(
                    "summarize main topics key points overview"
                )

                summary_prompt = ChatPromptTemplate.from_template("""
You are a document summarizer.
Based on the following document excerpts, provide:

1. 📌 Main Topic
2. 🔑 Key Points (bullet points)
3. 📊 Important Data/Numbers mentioned
4. 📝 Brief Summary (2-3 sentences)

Document excerpts:
{context}
""")
                chain = summary_prompt | llm | StrOutputParser()
                summary = chain.invoke({
                    "context": "\n\n".join([doc.page_content for doc in sample_docs])
                })
                st.session_state.doc_summary = summary

        if "doc_summary" in st.session_state and st.session_state.doc_summary:
            st.markdown("### 📝 Document Summary")
            st.markdown(st.session_state.doc_summary)

# ============================================================
# MODE 1: RESEARCH REPORT GENERATION
# ============================================================
if mode == "📋 Research Report Generation":

    if "report_result" not in st.session_state:
        st.session_state.report_result = None
    if "report_chat_history" not in st.session_state:
        st.session_state.report_chat_history = []

    st.subheader("🔍 Enter Your Research Topic")
    topic = st.text_input(
        label="Topic",
        placeholder="e.g. Latest trends in AI startups 2024",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("🚀 Generate Report", use_container_width=True)

    if run_button and topic:
        st.session_state.report_chat_history = []

        reader_agent = None
        if "retriever" in st.session_state:
            reader_agent = create_reader_agent(st.session_state.retriever)

        with st.status("🤖 Agents are working...", expanded=True) as status:
            st.write("🔎 Searcher Agent: Searching the web...")
            st.write("📄 Reader Agent: Scanning documents...")
            st.write("✍️ Writer Agent: Writing your report...")

            graph = build_graph(reader_agent)
            result = graph.invoke({
                "topic": topic,
                "web_findings": "",
                "doc_findings": "",
                "final_report": ""
            })

            status.update(label="✅ Report Ready!", state="complete")

        st.session_state.report_result = result
        st.session_state.report_topic = topic

    elif run_button and not topic:
        st.warning("⚠️ Please enter a research topic first!")

    if st.session_state.report_result:
        result = st.session_state.report_result

        st.subheader("📋 Research Report")
        st.markdown(result["final_report"])

        st.markdown("---")
        with st.expander("🌐 View Raw Web Findings"):
            st.write(result["web_findings"])

        if result["doc_findings"] != "No documents were uploaded by the user.":
            with st.expander("📄 View Raw Document Findings"):
                st.write(result["doc_findings"])

        # --- FOLLOW UP CHAT ---
        st.markdown("---")
        st.subheader("💬 Ask Follow-up Questions")

        for i, msg in enumerate(st.session_state.report_chat_history):
            with st.chat_message(msg["role"]):
                col1, col2 = st.columns([11, 1])
                with col1:
                    st.markdown(msg["content"])
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.report_chat_history.pop(i)
                        st.rerun()

        follow_up = st.chat_input("Ask anything about the report...")

        if follow_up:
            with st.chat_message("user"):
                st.markdown(follow_up)
            st.session_state.report_chat_history.append({"role": "user", "content": follow_up})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile"
                    )

                    # Build history for memory
                    history_text = ""
                    for msg in st.session_state.report_chat_history[:-1]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history_text += f"{role}: {msg['content']}\n"

                    prompt = ChatPromptTemplate.from_template("""
You are a research assistant. Answer the follow-up question based on the research context below.

Previous Conversation:
{history}

Research Topic: {topic}
Research Report: {report}
Web Findings: {web_findings}
Document Findings: {doc_findings}

Follow-up Question: {question}

Give a clear, detailed answer:
""")
                    chain = prompt | llm | StrOutputParser()

                    full_answer = ""
                    response_container = st.empty()

                    for chunk in chain.stream({
                        "history": history_text,
                        "topic": st.session_state.report_topic,
                        "report": result["final_report"],
                        "web_findings": result["web_findings"],
                        "doc_findings": result["doc_findings"],
                        "question": follow_up
                    }):
                        full_answer += chunk
                        response_container.markdown(full_answer + "▌")

                    response_container.markdown(full_answer)
                    st.session_state.report_chat_history.append({"role": "assistant", "content": full_answer})

        if st.session_state.report_chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Chat", use_container_width=True):
                    st.session_state.report_chat_history = []
                    st.rerun()
            with col2:
                pdf_bytes = export_chat_as_pdf(
                    st.session_state.report_chat_history,
                    "Research Follow-up Chat Export"
                )
                st.download_button(
                    label="📤 Export Chat as PDF",
                    data=bytes(pdf_bytes),
                    file_name="report_chat_export.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# ============================================================
# MODE 2: PDF Q&A CHAT
# ============================================================
elif mode == "💬 PDF Q&A Chat":

    if "retriever" not in st.session_state:
        st.info("👈 Upload your PDFs from the sidebar and click **Process PDFs** to start chatting!")

    else:
        st.subheader("💬 Chat with Your PDFs")

        col1, col2 = st.columns([3, 1])
        with col2:
            translate_to = st.selectbox(
                "🌐 Answer Language",
                ["English", "Hindi", "Tamil", "Telugu", "Spanish",
                 "French", "German", "Arabic", "Japanese", "Chinese"]
            )

        if "pdf_chat_history" not in st.session_state:
            st.session_state.pdf_chat_history = []

        for i, msg in enumerate(st.session_state.pdf_chat_history):
            with st.chat_message(msg["role"]):
                col1, col2 = st.columns([11, 1])
                with col1:
                    st.markdown(msg["content"])
                with col2:
                    if st.button("🗑️", key=f"pdf_delete_{i}"):
                        st.session_state.pdf_chat_history.pop(i)
                        st.rerun()

        question = st.chat_input("Ask a question about your PDFs...")

        if question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.pdf_chat_history.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile"
                    )

                    # Step 1: Rewrite query for better retrieval
                    rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer.
Rewrite the following question to be more specific and detailed for document retrieval.
Only return the rewritten question, nothing else.

Original question: {question}
Rewritten question:
""")
                    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
                    rewritten_question = rewrite_chain.invoke({"question": question})
                    print(f"🔄 Rewritten query: {rewritten_question}")

                    # Step 2: Retrieve docs manually for source display
                    retrieved_docs = st.session_state.retriever.invoke(rewritten_question)

                    def format_docs(docs):
                        return "\n\n".join([
                            f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
                            for doc in docs
                        ])

                    context_text = format_docs(retrieved_docs)

                    # Step 3: Build history for memory
                    history_text = ""
                    for msg in st.session_state.pdf_chat_history[:-1]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history_text += f"{role}: {msg['content']}\n"

                    # Step 4: Answer with context + memory
                    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions STRICTLY based on the provided document context.
Each piece of context includes which file it came from.
Do NOT make assumptions or use outside knowledge.
Only use information that is explicitly present in the context.
Never guess or approximate dates, amounts, or numbers.
If the answer is not clearly in the context, say "I couldn't find that information in the uploaded documents."

Previous Conversation:
{history}

Context:
{context}

Question:
{question}

Answer only using the context above.
IMPORTANT: Respond in {language} language only.
""")
                    chain = prompt | llm | StrOutputParser()

                    # Step 5: Stream the response
                    full_answer = ""
                    response_container = st.empty()

                    for chunk in chain.stream({
                        "context": context_text,
                        "question": rewritten_question,
                        "history": history_text,
                        "language": translate_to
                     }):
                        full_answer += chunk
                        response_container.markdown(full_answer + "▌")

                    response_container.markdown(full_answer)
                    st.session_state.pdf_chat_history.append({"role": "assistant", "content": full_answer})

                    # Step 6: Show sources
                    with st.expander("📄 View Sources"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            source = doc.metadata.get('source_file', 'unknown')
                            page = doc.metadata.get('page', 'unknown')
                            st.markdown(f"**Source {i}:** `{source}` — Page {page}")
                            st.caption(doc.page_content[:300] + "...")
                            st.markdown("---")

        if st.session_state.get("pdf_chat_history"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Chat", use_container_width=True):
                    st.session_state.pdf_chat_history = []
                    st.rerun()
            with col2:
                pdf_bytes = export_chat_as_pdf(
                    st.session_state.pdf_chat_history,
                    "PDF Q&A Chat Export"
                )
                st.download_button(
                    label="📤 Export Chat as PDF",
                    data=bytes(pdf_bytes),
                    file_name="chat_export.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
# ============================================================
# MODE 3: COMPARE TWO PDFs
# ============================================================
elif mode == "🔄 Compare PDFs":

    st.subheader("🔄 Compare Two PDFs Side by Side")
    st.markdown("Upload exactly **2 PDFs** in the sidebar and click Process PDFs first.")

    if "retriever" not in st.session_state:
        st.info("👈 Upload 2 PDFs from the sidebar and click **Process PDFs** to start comparing!")

    elif len(st.session_state.pdf_names) < 2:
        st.warning("⚠️ Please upload at least 2 PDFs to compare!")

    else:
        compare_topic = st.text_input(
            "What do you want to compare?",
            placeholder="e.g. payment terms, key dates, total amounts"
        )

        if st.button("🔄 Compare Now", use_container_width=False):
            if not compare_topic:
                st.warning("⚠️ Please enter what you want to compare!")
            else:
                with st.spinner("Comparing documents..."):
                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile"
                    )

                    # Get relevant chunks from both docs
                    retrieved_docs = st.session_state.retriever.invoke(compare_topic)

                    # Separate chunks by source file
                    doc1_chunks = [d for d in retrieved_docs
                                   if d.metadata.get("source_file") == st.session_state.pdf_names[0]]
                    doc2_chunks = [d for d in retrieved_docs
                                   if d.metadata.get("source_file") == st.session_state.pdf_names[1]]

                    doc1_text = "\n".join([d.page_content for d in doc1_chunks]) or "No relevant content found"
                    doc2_text = "\n".join([d.page_content for d in doc2_chunks]) or "No relevant content found"

                    compare_prompt = ChatPromptTemplate.from_template("""
You are a document comparison expert.
Compare the two documents below on the topic: {topic}

Document 1 ({doc1_name}):
{doc1_content}

Document 2 ({doc2_name}):
{doc2_content}

Provide a detailed comparison with:
1. 📄 Document 1 Summary on this topic
2. 📄 Document 2 Summary on this topic
3. ✅ Similarities
4. ❌ Differences
5. 📊 Key Takeaway
""")
                    chain = compare_prompt | llm | StrOutputParser()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### 📄 {st.session_state.pdf_names[0]}")
                        st.markdown(doc1_text[:1000] + "..." if len(doc1_text) > 1000 else doc1_text)

                    with col2:
                        st.markdown(f"### 📄 {st.session_state.pdf_names[1]}")
                        st.markdown(doc2_text[:1000] + "..." if len(doc2_text) > 1000 else doc2_text)

                    st.markdown("---")
                    st.subheader("📊 AI Comparison Analysis")

                    full_comparison = ""
                    response_container = st.empty()

                    for chunk in chain.stream({
                        "topic": compare_topic,
                        "doc1_name": st.session_state.pdf_names[0],
                        "doc2_name": st.session_state.pdf_names[1],
                        "doc1_content": doc1_text,
                        "doc2_content": doc2_text
                    }):
                        full_comparison += chunk
                        response_container.markdown(full_comparison + "▌")

                    response_container.markdown(full_comparison)