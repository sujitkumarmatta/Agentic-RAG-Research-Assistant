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
from fpdf import FPDF
from database import init_db, save_chat, load_chats, load_chat_by_id, save_report, load_reports, load_report_by_id, delete_chat, delete_report
from url_loader import load_from_url
import speech_recognition as sr
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import hashlib
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from reflection import create_reflection_agent
from ragas_eval import evaluate_rag_quality
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# --- INIT DATABASE ---
init_db()

# --- HELPER FUNCTIONS ---
def get_cache_key(question, pdf_names):
    key = f"{question}_{sorted(pdf_names)}"
    return hashlib.md5(key.encode()).hexdigest()

def get_cached_answer(cache_key):
    cache = st.session_state.get("answer_cache", {})
    return cache.get(cache_key)

def save_to_cache(cache_key, answer):
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}
    st.session_state.answer_cache[cache_key] = answer

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
        content = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, content)
        pdf.ln(3)
    return pdf.output()

def listen_to_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=10)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        return None

def extract_numbers_from_text(text):
    pattern = r'(\w[\w\s]*?):\s*([\d,\.]+)'
    matches = re.findall(pattern, text)
    return {k.strip(): float(v.replace(',', '')) for k, v in matches if len(k) < 50}

def send_email(to_email, subject, body, smtp_user, smtp_pass):
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

def safe_calculate(expression: str):
    """Safely evaluate mathematical expressions"""
    try:
        # Only allow safe math operations
        allowed = {
            'abs': abs, 'round': round,
            'min': min, 'max': max, 'sum': sum
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except:
        return "Could not calculate"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "stats" not in st.session_state:
    st.session_state.stats = {
        "questions_asked": 0,
        "docs_uploaded": 0,
        "reports_generated": 0,
        "comparisons_done": 0
    }
if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "report_chat_history" not in st.session_state:
    st.session_state.report_chat_history = []
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []

# --- HEADER ---
st.title("🔬 Multi-Agent Research Assistant")

# --- AI PERSONA SELECTOR ---
with st.expander("🤖 Customize AI Persona & Language", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        persona_name = st.text_input("Assistant Name", value="Aria")
    with col2:
        persona_style = st.selectbox("Personality Style", [
            "Professional & Formal",
            "Friendly & Casual",
            "Concise & Direct",
            "Detailed & Academic",
            "Enthusiastic & Encouraging"
        ])
    with col3:
        persona_role = st.selectbox("Expert Role", [
            "Research Analyst",
            "Legal Expert",
            "Financial Advisor",
            "Medical Consultant",
            "Tech Expert",
            "General Assistant"
        ])
    with col4:
        translate_to = st.selectbox(
            "🌐 Response Language",
            ["English", "Hindi", "Tamil", "Telugu", "Spanish",
             "French", "German", "Arabic", "Japanese", "Chinese"]
        )

persona_prompt = f"Your name is {persona_name}. You are a {persona_role}. Your communication style is {persona_style}."

st.markdown("---")

# --- MODE SELECTOR ---
mode = st.radio(
    "Choose Mode:",
    ["📋 Research Report Generation", "💬 PDF Q&A Chat",
     "🔄 Compare PDFs", "💭 Chat", "📚 Chat History", "📊 Analytics"],
    horizontal=True
)

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    # --- USAGE STATS ---
    st.markdown("### 📈 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("❓ Questions", st.session_state.stats["questions_asked"])
        st.metric("📋 Reports", st.session_state.stats["reports_generated"])
    with col2:
        st.metric("📄 Docs", st.session_state.stats["docs_uploaded"])
        st.metric("🔄 Compares", st.session_state.stats["comparisons_done"])

    st.markdown("---")
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # --- URL LOADER ---
    st.markdown("### 🔗 Or Load from URL")
    url_input = st.text_input("Paste a website URL", placeholder="https://example.com/article")
    load_url_btn = st.button("🔗 Load URL", use_container_width=True)

    if uploaded_files or load_url_btn:
        if st.button("📥 Process Sources", use_container_width=True) or load_url_btn:
            all_chunks = []
            pdf_names = []

            # Process PDFs
            if uploaded_files:
                with st.spinner("Reading PDFs..."):
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
                            st.sidebar.write(f"⚠️ {uploaded_file.name} — skipped")

            # Process URL
            if load_url_btn and url_input:
                with st.spinner(f"Loading {url_input}..."):
                    url_chunks = load_from_url(url_input)
                    if url_chunks:
                        all_chunks.extend(url_chunks)
                        pdf_names.append(url_input)
                        st.sidebar.write(f"✅ URL loaded — {len(url_chunks)} chunks")
                    else:
                        st.sidebar.error("⚠️ Could not load URL")

            if not all_chunks:
                st.error("⚠️ No content could be extracted.")
            else:
                vector_store, stored_chunks = create_vector_store(all_chunks)
                retriever = get_retriever(vector_store, stored_chunks)
                st.session_state.retriever = retriever
                st.session_state.pdf_names = pdf_names
                st.session_state.pdf_chat_history = []
                st.session_state.total_chunks = len(all_chunks)
                st.session_state.doc_summary = None
                st.session_state.stats["docs_uploaded"] += len(pdf_names)
                st.success(f"✅ {len(pdf_names)} source(s) indexed! ({len(all_chunks)} chunks)")

    if "retriever" in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Loaded Sources")
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
                from agents.writer import write_summary
                summary = write_summary(
                    llm,
                    "\n\n".join([doc.page_content for doc in sample_docs]),
                    translate_to
                )
                st.session_state.doc_summary = summary

        if "doc_summary" in st.session_state and st.session_state.doc_summary:
            st.markdown("### 📝 Summary")
            st.markdown(st.session_state.doc_summary)

# ============================================================
# MODE 1: RESEARCH REPORT GENERATION
# ============================================================
if mode == "📋 Research Report Generation":

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
            st.write("📋 Planner: Creating research plan...")
            st.write("🔀 Running Searcher + Reader in parallel...")
            st.write("✍️ Writer Agent: Writing your report...")
            st.write("🛡️ Fact Checker: Verifying claims...")

            graph = build_graph(reader_agent)
            result = graph.invoke({
                "topic": topic,
                "plan": "",
                "web_findings": "",
                "doc_findings": "",
                "final_report": "",
                "fact_check": "",
                "language": translate_to,
                "has_documents": "retriever" in st.session_state
            })
            status.update(label="✅ Report Ready!", state="complete")

        st.session_state.report_result = result
        st.session_state.report_topic = topic
        st.session_state.stats["reports_generated"] += 1
        save_report(topic, result["final_report"], result["web_findings"], result["doc_findings"])

    elif run_button and not topic:
        st.warning("⚠️ Please enter a research topic first!")

    if st.session_state.report_result:
        result = st.session_state.report_result

        if result.get("plan"):
                with st.expander("📋 View Research Plan"):
                    st.markdown(result["plan"])

        st.subheader("📋 Research Report")
        st.markdown(result["final_report"])

        st.markdown("---")
        with st.expander("🌐 View Raw Web Findings"):
            st.write(result["web_findings"])

        if result["doc_findings"] != "No documents were uploaded by the user.":
            with st.expander("📄 View Raw Document Findings"):
                st.write(result["doc_findings"])
        # Show fact check results
        if result.get("fact_check"):
            with st.expander("🛡️ View Fact Check Report"):
                st.markdown(result["fact_check"])

        # --- EMAIL REPORT ---
        with st.expander("📧 Email This Report"):
            col1, col2 = st.columns(2)
            with col1:
                to_email = st.text_input("Recipient Email")
                smtp_user = st.text_input("Your Gmail", placeholder="yourname@gmail.com")
            with col2:
                smtp_pass = st.text_input("App Password", type="password",
                    help="Use Gmail App Password, not your regular password")
                email_subject = st.text_input("Subject", value=f"Research Report: {st.session_state.get('report_topic', '')}")

            if st.button("📧 Send Email"):
                if to_email and smtp_user and smtp_pass:
                    try:
                        send_email(to_email, email_subject, result["final_report"], smtp_user, smtp_pass)
                        st.success("✅ Email sent successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")
                else:
                    st.warning("Please fill all email fields!")

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
            st.session_state.stats["questions_asked"] += 1

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile"
                    )

                    history_text = ""
                    for msg in st.session_state.report_chat_history[:-1]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history_text += f"{role}: {msg['content']}\n"

                    # Pre-calculate if question involves numbers
                    calc_context = ""
                    numbers_in_question = re.findall(r'\d+', follow_up)
                    if numbers_in_question and any(
                        word in question.lower() for word in
                        ['above', 'below', 'greater', 'less', 'more', 'than', 'over', 'under', 'between']
                    ):
                        calc_context = f"\nNOTE: Verify each number mathematically. Numbers in question: {numbers_in_question}"

                    prompt = ChatPromptTemplate.from_template("""
{persona}

You are a precise financial data analyst and helpful assistant.
Answer STRICTLY based on the provided document context.

CRITICAL MATH RULES:
- Every number in the document is already in plain format (commas removed)
- For comparisons like "above 70000", verify EACH number mathematically
- 68147 > 70000 is FALSE — do not include it
- 825517 > 70000 is TRUE — include it
- Double check every single number before including in your answer
- Never include a number unless it strictly satisfies the condition

{calc_note}

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

                    full_answer = ""
                    response_container = st.empty()

                    for chunk in chain.stream({
                        "persona": persona_prompt,
                        "history": history_text,
                        "topic": st.session_state.report_topic,
                        "report": result["final_report"],
                        "web_findings": result["web_findings"],
                        "doc_findings": result["doc_findings"],
                        "question": follow_up,
                        "language": translate_to,
                        "calc_note": calc_context

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

            if st.button("💾 Save Chat to History"):
                save_chat(
                    f"Report: {st.session_state.report_topic}",
                    "report",
                    st.session_state.report_chat_history
                )
                st.success("✅ Chat saved!")

# ============================================================
# MODE 2: PDF Q&A CHAT
# ============================================================
elif mode == "💬 PDF Q&A Chat":

    if "retriever" not in st.session_state:
        st.info("👈 Upload your PDFs from the sidebar and click **Process Sources** to start chatting!")
    else:
        st.subheader("💬 Chat with Your Documents")

        

        for i, msg in enumerate(st.session_state.pdf_chat_history):
            with st.chat_message(msg["role"]):
                col1, col2 = st.columns([11, 1])
                with col1:
                    st.markdown(msg["content"])
                with col2:
                    if st.button("🗑️", key=f"pdf_delete_{i}"):
                        st.session_state.pdf_chat_history.pop(i)
                        st.rerun()

        # --- VOICE INPUT ---
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🎤 Voice Input", use_container_width=True):
                voice_text = listen_to_voice()
                if voice_text:
                    st.session_state.voice_input = voice_text
                    st.success(f"✅ Heard: {voice_text}")
                else:
                    st.error("❌ Could not understand. Try again.")

        voice_prefill = st.session_state.get("voice_input", "")
        question = st.chat_input("Ask a question about your documents...")

        # Use voice input if available
        if not question and voice_prefill:
            question = voice_prefill
            st.session_state.voice_input = ""

        if question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.pdf_chat_history.append({"role": "user", "content": question})
            st.session_state.stats["questions_asked"] += 1

            with st.chat_message("assistant"):
                cache_key = get_cache_key(question, st.session_state.pdf_names)
                cached = get_cached_answer(cache_key)

                if cached:
                    st.markdown(cached)
                    st.caption("⚡ Cached response")
                    st.session_state.pdf_chat_history.append({"role": "assistant", "content": cached})
                else:
                    with st.spinner("Thinking..."):
                        llm = ChatGroq(
                            api_key=os.getenv("GROQ_API_KEY"),
                            model_name="llama-3.3-70b-versatile",
                            temperature=0.3
                        )

                        # Rewrite query
                        rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer.
Rewrite the following question to be more specific and detailed for document retrieval.
Only return the rewritten question, nothing else.

Original question: {question}
Rewritten question:
""")
                        rewrite_chain = rewrite_prompt | llm | StrOutputParser()
                        rewritten_question = rewrite_chain.invoke({"question": question})

                        # Retrieve docs
                        retrieved_docs = st.session_state.retriever.invoke(rewritten_question)

                        def format_docs(docs):
                            return "\n\n".join([
                                f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
                                for doc in docs
                            ])

                        context_text = format_docs(retrieved_docs)

                        # Build history
                        history_text = ""
                        for msg in st.session_state.pdf_chat_history[:-1]:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            history_text += f"{role}: {msg['content']}\n"

                        # Answer
                        prompt = ChatPromptTemplate.from_template("""
{persona}

You are a precise analyst. You MUST think step by step before answering.

CHAIN OF THOUGHT PROCESS:
STEP 1 — UNDERSTAND: What exactly is the question asking?
STEP 2 — EXTRACT: What relevant information exists in the context?
STEP 3 — VERIFY: If numbers involved, check each one mathematically:
         - Write out: "Is [number] satisfying condition [X]? Yes/No"
         - Example: "Is 68147 > 70000? No — exclude"
         - Example: "Is 825517 > 70000? Yes — include"
STEP 4 — ANSWER: Form precise answer using only verified information

STRICT RULES:
- Never include a number unless Step 3 confirms it satisfies condition
- Never guess or assume
- If not in context say "I couldn't find that in the documents"

{calc_note}

Previous Conversation:
{history}

Context:
{context}

Question:
{question}

Show your Step 1-4 thinking, then give Final Answer.
IMPORTANT: Respond in {language} language only.
""")
                        chain = prompt | llm | StrOutputParser()

                        full_answer = ""
                        response_container = st.empty()

                        for chunk in chain.stream({
                            "persona": persona_prompt,
                            "context": context_text,
                            "question": rewritten_question,
                            "history": history_text,
                            "language": translate_to
                        }):
                            full_answer += chunk
                            response_container.markdown(full_answer + "▌")

                        # Self-reflection
                        with st.spinner("🔄 Self-reflecting on answer..."):
                            reflector = create_reflection_agent()
                            improved_answer, critique = reflector(
                                question,
                                full_answer,
                                context_text
                            )

                        if improved_answer != full_answer:
                            response_container.markdown(improved_answer)
                            full_answer = improved_answer
                            with st.expander("🔄 View Self-Reflection Critique"):
                                st.markdown(critique)

                        st.session_state.pdf_chat_history.append({"role": "assistant", "content": full_answer})
                        save_to_cache(cache_key, full_answer)

                        # Confidence score
                        confidence_prompt = ChatPromptTemplate.from_template("""
Rate confidence in the answer on a scale 0-100 based on context support.
Reply with ONLY a number.

Answer: {answer}
Context: {context}
""")
                        confidence_chain = confidence_prompt | llm | StrOutputParser()
                        try:
                            score_text = confidence_chain.invoke({
                                "answer": full_answer,
                                "context": context_text
                            })
                            score = int("".join(filter(str.isdigit, score_text)))
                            score = min(100, max(0, score))
                            color = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
                            label = "High" if score >= 80 else "Medium" if score >= 50 else "Low"
                            st.markdown(f"**Confidence:** {color} {label} ({score}%)")
                            st.progress(score / 100)
                        except:
                            pass

                        # Auto chart generation
                        numbers = extract_numbers_from_text(full_answer)
                        if len(numbers) >= 2:
                            with st.expander("📊 Auto Generated Chart"):
                                fig = px.bar(
                                    x=list(numbers.keys()),
                                    y=list(numbers.values()),
                                    title="Data from Answer",
                                    labels={"x": "Category", "y": "Value"}
                                )
                                fig.update_layout(
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font_color="white"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        # Show sources
                        with st.expander("📄 View Sources"):
                            for i, doc in enumerate(retrieved_docs, 1):
                                source = doc.metadata.get('source_file', 'unknown')
                                page = doc.metadata.get('page', 'unknown')
                                st.markdown(f"**Source {i}:** `{source}` — Page {page}")
                                st.caption(doc.page_content[:300] + "...")
                                st.markdown("---")

        # RAGAS Evaluation
        if st.session_state.get("pdf_chat_history") and len(st.session_state.pdf_chat_history) >= 2:
            if st.button("📊 Evaluate RAG Quality (RAGAS)", use_container_width=True):
                with st.spinner("Running RAGAS evaluation..."):
                    # Extract Q&A pairs
                    questions = []
                    answers = []
                    contexts = []

                    history = st.session_state.pdf_chat_history
                    for i in range(0, len(history)-1, 2):
                        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
                            questions.append(history[i]["content"])
                            answers.append(history[i+1]["content"])
                            contexts.append(context_text if 'context_text' in dir() else "")

                    if questions:
                        scores = evaluate_rag_quality(questions, answers, contexts)
                        if scores:
                            st.markdown("### 📊 RAGAS Evaluation Results")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("🎯 Faithfulness", f"{scores['faithfulness']:.0%}")
                            with col2:
                                st.metric("💡 Relevancy", f"{scores['answer_relevancy']:.0%}")
                            with col3:
                                st.metric("🎯 Precision", f"{scores['context_precision']:.0%}")
                            with col4:
                                st.metric("📚 Recall", f"{scores['context_recall']:.0%}")
                            with col5:
                                st.metric("⭐ Overall", f"{scores['overall']:.0%}")

        if st.session_state.get("pdf_chat_history"):
            col1, col2, col3 = st.columns(3)
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
                    label="📤 Export as PDF",
                    data=bytes(pdf_bytes),
                    file_name="chat_export.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col3:
                if st.button("💾 Save to History", use_container_width=True):
                    save_chat(
                        f"PDF Chat: {', '.join(st.session_state.pdf_names[:2])}",
                        "pdf_qa",
                        st.session_state.pdf_chat_history
                    )
                    st.success("✅ Saved!")
# ============================================================
# MODE: SIMPLE CHAT
# ============================================================
elif mode == "💭 Chat":

    st.subheader(f"💭 Chat with {persona_name}")
    st.caption(f"*{persona_role} • {persona_style} • Responding in {translate_to}*")

    if "simple_chat_history" not in st.session_state:
        st.session_state.simple_chat_history = []

    # Display chat history
    for i, msg in enumerate(st.session_state.simple_chat_history):
        with st.chat_message(msg["role"]):
            col1, col2 = st.columns([11, 1])
            with col1:
                st.markdown(msg["content"])
            with col2:
                if st.button("🗑️", key=f"simple_delete_{i}"):
                    st.session_state.simple_chat_history.pop(i)
                    st.rerun()

    # Voice input
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🎤 Voice", use_container_width=True):
            voice_text = listen_to_voice()
            if voice_text:
                st.session_state.simple_voice_input = voice_text
                st.success(f"✅ Heard: {voice_text}")
            else:
                st.error("❌ Could not understand. Try again.")

    voice_prefill = st.session_state.get("simple_voice_input", "")
    question = st.chat_input(f"Chat with {persona_name}...")

    if not question and voice_prefill:
        question = voice_prefill
        st.session_state.simple_voice_input = ""

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.simple_chat_history.append({"role": "user", "content": question})
        st.session_state.stats["questions_asked"] += 1

        with st.chat_message("assistant"):
            with st.spinner(f"{persona_name} is thinking..."):
                llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile"
                )

                # Build history
                history_text = ""
                for msg in st.session_state.simple_chat_history[:-1]:
                    role = "User" if msg["role"] == "user" else persona_name
                    history_text += f"{role}: {msg['content']}\n"

                prompt = ChatPromptTemplate.from_template("""
{persona}

You are having a conversation with a user.
Be helpful, engaging and stay in character.

Previous Conversation:
{history}

User: {question}

IMPORTANT: Respond in {language} language only.
""")
                chain = prompt | llm | StrOutputParser()

                full_answer = ""
                response_container = st.empty()

                for chunk in chain.stream({
                    "persona": persona_prompt,
                    "history": history_text,
                    "question": question,
                    "language": translate_to
                }):
                    full_answer += chunk
                    response_container.markdown(full_answer + "▌")

                response_container.markdown(full_answer)
                st.session_state.simple_chat_history.append({"role": "assistant", "content": full_answer})

    if st.session_state.get("simple_chat_history"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.simple_chat_history = []
                st.rerun()
        with col2:
            pdf_bytes = export_chat_as_pdf(
                st.session_state.simple_chat_history,
                f"Chat with {persona_name}"
            )
            st.download_button(
                label="📤 Export as PDF",
                data=bytes(pdf_bytes),
                file_name="chat_export.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with col3:
            if st.button("💾 Save to History", use_container_width=True):
                save_chat(
                    f"Chat: {st.session_state.simple_chat_history[0]['content'][:30]}...",
                    "simple_chat",
                    st.session_state.simple_chat_history
                )
                st.success("✅ Saved!")

# ============================================================
# MODE 3: COMPARE PDFs
# ============================================================
elif mode == "🔄 Compare PDFs":

    st.subheader("🔄 Compare Two Documents Side by Side")

    if "retriever" not in st.session_state:
        st.info("👈 Upload 2 PDFs from the sidebar and click **Process Sources**!")
    elif len(st.session_state.pdf_names) < 2:
        st.warning("⚠️ Please upload at least 2 PDFs to compare!")
    else:
        compare_topic = st.text_input(
            "What do you want to compare?",
            placeholder="e.g. payment terms, key dates, total amounts"
        )

        if st.button("🔄 Compare Now"):
            if not compare_topic:
                st.warning("⚠️ Please enter what you want to compare!")
            else:
                with st.spinner("Comparing documents..."):
                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.3-70b-versatile"
                    )

                    retrieved_docs = st.session_state.retriever.invoke(compare_topic)
                    doc1_chunks = [d for d in retrieved_docs
                                   if d.metadata.get("source_file") == st.session_state.pdf_names[0]]
                    doc2_chunks = [d for d in retrieved_docs
                                   if d.metadata.get("source_file") == st.session_state.pdf_names[1]]

                    doc1_text = "\n".join([d.page_content for d in doc1_chunks]) or "No relevant content found"
                    doc2_text = "\n".join([d.page_content for d in doc2_chunks]) or "No relevant content found"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### 📄 {st.session_state.pdf_names[0]}")
                        st.markdown(doc1_text[:1000] + "..." if len(doc1_text) > 1000 else doc1_text)
                    with col2:
                        st.markdown(f"### 📄 {st.session_state.pdf_names[1]}")
                        st.markdown(doc2_text[:1000] + "..." if len(doc2_text) > 1000 else doc2_text)

                    st.markdown("---")
                    st.subheader("📊 AI Comparison Analysis")
                    
                    from agents.writer import write_comparison
                    full_comparison = write_comparison(
                        llm,
                        compare_topic,
                        st.session_state.pdf_names[0],
                        doc1_text,
                        st.session_state.pdf_names[1],
                        doc2_text,
                        translate_to
                    )

                    response_container = st.empty()
                    response_container.markdown(full_comparison)
                    st.session_state.stats["comparisons_done"] += 1

# ============================================================
# MODE 4: CHAT HISTORY
# ============================================================
elif mode == "📚 Chat History":

    st.subheader("📚 Saved Chat History")

    tab1, tab2 = st.tabs(["💬 Saved Chats", "📋 Saved Reports"])

    with tab1:
        chats = load_chats()
        if not chats:
            st.info("No saved chats yet. Save a chat from the Q&A or Report mode!")
        else:
            for chat_id, name, chat_mode, created_at in chats:
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.markdown(f"**{name}**")
                    st.caption(f"Mode: {chat_mode} | {created_at[:16]}")
                with col2:
                    if st.button("📖 Load", key=f"load_chat_{chat_id}"):
                        messages = load_chat_by_id(chat_id)
                        st.session_state.loaded_chat = messages
                        st.session_state.loaded_chat_name = name
                with col3:
                    if st.button("🗑️", key=f"del_chat_{chat_id}"):
                        delete_chat(chat_id)
                        st.rerun()

            if "loaded_chat" in st.session_state:
                st.markdown("---")
                st.subheader(f"📖 {st.session_state.loaded_chat_name}")
                for msg in st.session_state.loaded_chat:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

    with tab2:
        reports = load_reports()
        if not reports:
            st.info("No saved reports yet. Generate a report first!")
        else:
            for rep_id, topic, created_at in reports:
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.markdown(f"**{topic}**")
                    st.caption(created_at[:16])
                with col2:
                    if st.button("📖 View", key=f"load_rep_{rep_id}"):
                        row = load_report_by_id(rep_id)
                        st.session_state.loaded_report = row
                with col3:
                    if st.button("🗑️", key=f"del_rep_{rep_id}"):
                        delete_report(rep_id)
                        st.rerun()

            if "loaded_report" in st.session_state:
                st.markdown("---")
                row = st.session_state.loaded_report
                st.subheader(f"📋 {row[1]}")
                st.markdown(row[2])
                with st.expander("🌐 Web Findings"):
                    st.write(row[3])
                with st.expander("📄 Doc Findings"):
                    st.write(row[4])

# ============================================================
# MODE 5: ANALYTICS
# ============================================================
elif mode == "📊 Analytics":

    st.subheader("📊 Usage Analytics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("❓ Questions Asked", st.session_state.stats["questions_asked"])
    with col2:
        st.metric("📄 Docs Uploaded", st.session_state.stats["docs_uploaded"])
    with col3:
        st.metric("📋 Reports Generated", st.session_state.stats["reports_generated"])
    with col4:
        st.metric("🔄 Comparisons Done", st.session_state.stats["comparisons_done"])

    st.markdown("---")

    # Activity chart
    stats_data = {
        "Activity": ["Questions", "Docs", "Reports", "Comparisons"],
        "Count": [
            st.session_state.stats["questions_asked"],
            st.session_state.stats["docs_uploaded"],
            st.session_state.stats["reports_generated"],
            st.session_state.stats["comparisons_done"]
        ]
    }

    fig = px.bar(
        stats_data,
        x="Activity",
        y="Count",
        title="Session Activity Overview",
        color="Activity",
        color_discrete_sequence=["#00d2ff", "#7b2ff7", "#00ff88", "#ff6b6b"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Saved data stats
    st.markdown("---")
    st.subheader("💾 Persistent Storage Stats")
    saved_chats = load_chats()
    saved_reports = load_reports()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Total Saved Chats", len(saved_chats))
    with col2:
        st.metric("📋 Total Saved Reports", len(saved_reports))