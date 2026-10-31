import os
import json
import requests
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON as RichJSON

console = Console()

# --- CONFIGURATION ---
DATA_PATH = "data"
DB_PATH = "chroma_db"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
JUDGE_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# ==========================================
# PRE-FLIGHT CHECK
# ==========================================
def check_ollama():
    """Verify Ollama is running and required models are available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "[ERROR] Cannot connect to Ollama. "
            "Please start it with: ollama serve"
        )
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"[ERROR] Ollama health check failed: {e}")

    available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
    missing = [m for m in [OLLAMA_MODEL, EMBEDDING_MODEL] if m not in available]
    if missing:
        for m in missing:
            console.print(f"[red][ERROR] Model '{m}' not found. Run: ollama pull {m}[/red]")
        raise SystemExit("Please pull the missing models and try again.")

    console.print(f"[green][OK] Ollama running — models: {', '.join(available)}[/green]")

# ==========================================
# GUARDRAILS MODULE
# ==========================================
def input_guardrail(user_query: str) -> bool:
    """Checks user input for malicious patterns or unsafe content."""
    query_lower = user_query.lower()
    forbidden_keywords = ["ignore previous instructions", "system prompt", "hack"]
    for word in forbidden_keywords:
        if word in query_lower:
            console.print(f"[yellow][Guardrail] Blocked keyword: '{word}'[/yellow]")
            return False
    if len(user_query.strip()) < 3:
        console.print("[yellow][Guardrail] Query too short.[/yellow]")
        return False
    return True

def output_guardrail(response_text: str) -> str:
    """Inspects and cleans the generated LLM response."""
    if not response_text or len(response_text.strip()) == 0:
        return "Guardrail Error: The model returned an empty response."
    return response_text.replace("<|im_start|>", "").replace("<|im_end|>", "")

# ==========================================
# FAITHFULNESS JUDGE (LLM-AS-JUDGE)
# ==========================================
JUDGE_SYSTEM_PROMPT = (
    "You are a strict factual-consistency judge for a RAG system. "
    "You are given a retrieval context and an answer that was generated from it. "
    "Your task is to decide whether the answer is FAITHFUL, meaning every claim in the "
    "answer is supported by the provided context. "
    "If the answer contains information that is not present in the context, it is unfaithful. "
    "Respond with strict JSON only in the format: "
    '{{"faithful": true, "score": 0.0}} '
    "where score is a float between 0 and 1 (1.0 = fully grounded in the context)."
)

def judge_faithfulness(context: str, answer: str, llm) -> dict:
    """Uses an LLM judge to score how faithfully the answer is grounded in the context."""
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ])
    raw_judgment = (judge_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "answer": answer,
    })

    try:
        parsed = json.loads(raw_judgment.strip())
        score = float(parsed.get("score") or (1.0 if parsed.get("faithful") else 0.0))
        return {"faithful": bool(parsed.get("faithful", True)), "score": score}
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback heuristic if the judge response is not valid JSON
        judgment_lower = raw_judgment.lower()
        faithful = not any(
            w in judgment_lower
            for w in ["unfaithful", "not faithful", "not grounded", "hallucin"]
        )
        return {"faithful": faithful, "score": 1.0 if faithful else 0.0}

# ==========================================
# RAG PIPELINE SETUP
# ==========================================
def initialize_vector_db():
    """Loads PDFs, chunks them, embeds, and saves to local Chroma vector store."""
    os.makedirs(DATA_PATH, exist_ok=True)

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        console.print(
            Panel(
                "[bold yellow]No PDF files found in the 'data/' folder.\n\n"
                "Please add one or more PDF files to:[/bold yellow]\n"
                f"[cyan]{os.path.abspath(DATA_PATH)}[/cyan]",
                title="[red]Setup Required[/red]",
                expand=False,
            )
        )
        raise SystemExit("Add PDFs to the data/ folder and run again.")

    console.print(f"[bold]Found {len(pdf_files)} PDF(s):[/bold] {', '.join(pdf_files)}")

    with console.status("[bold green]Loading PDF pages...", spinner="dots"):
        loader = PyPDFDirectoryLoader(DATA_PATH)
        docs = loader.load()
    console.print(f"[green]✓[/green] Loaded {len(docs)} pages from {len(pdf_files)} file(s)")

    with console.status("[bold green]Splitting into chunks...", spinner="dots"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
    console.print(f"[green]✓[/green] Created {len(chunks)} chunks")

    with console.status("[bold green]Generating embeddings & saving to ChromaDB...", spinner="dots"):
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=DB_PATH
        )
    console.print(f"[green]✓[/green] Embeddings saved to ChromaDB")

    return vector_store


def get_or_create_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        with console.status("[bold green]Loading existing ChromaDB index...", spinner="dots"):
            db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        console.print("[green]✓[/green] Loaded existing ChromaDB index")
        return db
    else:
        return initialize_vector_db()

# ==========================================
# STREAMING QUERY + JSON RESULT
# ==========================================
def query_rag(retriever, prompt, llm, user_input: str) -> dict:
    """Retrieves docs, streams the LLM answer, and returns a structured JSON result."""

    # 1. Retrieve relevant chunks
    with console.status("[bold green]Retrieving relevant chunks...", spinner="dots"):
        source_docs = retriever.invoke(user_input)

    # 2. Build context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(source_docs)

    # 3. Stream the answer token-by-token
    answer_chain = prompt | llm | StrOutputParser()
    console.print("\n[bold cyan]Answer:[/bold cyan] ", end="")
    answer_chunks = []
    for chunk in answer_chain.stream({"context": context, "input": user_input}):
        print(chunk, end="", flush=True)
        answer_chunks.append(chunk)
    print()  # newline after streaming

    raw_answer = "".join(answer_chunks)

    # 4. Apply output guardrail
    final_answer = output_guardrail(raw_answer)

    # 5. LLM-as-judge faithfulness check
    judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0)
    judgment = judge_faithfulness(context, final_answer, judge_llm)
    if not judgment["faithful"]:
        console.print("[yellow][Guardrail] Answer flagged as unfaithful by the judge.[/yellow]")

    # 6. Build deduplicated source metadata
    seen = set()
    sources = []
    for doc in source_docs:
        meta = doc.metadata
        file_name = os.path.basename(meta.get("source", "unknown"))
        page_num = meta.get("page", 0) + 1  # PyPDF is 0-indexed; show 1-indexed
        key = (file_name, page_num)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": file_name,
                "page": page_num,
                "excerpt": doc.page_content[:250].strip() + (
                    "..." if len(doc.page_content) > 250 else ""
                ),
            })

    return {
        "answer": final_answer,
        "sources": sources,
        "faithfulness": judgment,
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    console.rule("[bold blue]Local RAG with Guardrails[/bold blue]")
    check_ollama()
    vector_store = get_or_create_db()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the following retrieved context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    console.print("\n[bold green]Ready! Ask questions about your PDFs. Type 'exit' to quit.[/bold green]\n")

    while True:
        try:
            user_input = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        # Input guardrail
        if not input_guardrail(user_input):
            console.print("[red]Blocked by input guardrail.[/red]\n")
            continue

        try:
            result = query_rag(retriever, prompt, llm, user_input)

            # Print structured JSON result
            console.print("\n[bold magenta]Structured Result (JSON):[/bold magenta]")
            console.print(RichJSON(json.dumps(result, indent=2)))
            console.rule()

        except Exception as e:
            console.print(f"[red]Error during retrieval or generation: {e}[/red]")


if __name__ == "__main__":
    main()