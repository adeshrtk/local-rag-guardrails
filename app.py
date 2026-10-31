import os
import requests
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DATA_PATH = "data"
DB_PATH = "chroma_db"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
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
    missing = []
    for model in [OLLAMA_MODEL, EMBEDDING_MODEL]:
        if model not in available:
            missing.append(model)
    if missing:
        for m in missing:
            print(f"[ERROR] Model '{m}' not found. Run: ollama pull {m}")
        raise SystemExit("Please pull the missing models and try again.")

    print(f"[OK] Ollama is running. Models available: {', '.join(available)}")


# ==========================================
# GUARDRAILS MODULE
# ==========================================
def input_guardrail(user_query: str) -> bool:
    """Checks user input for malicious patterns or unsafe content."""
    
    query_lower = user_query.lower()
    
    # Simple heuristic checks (extendable with NeMo Guardrails or Llama Guard if preferred)
    forbidden_keywords = ["ignore previous instructions", "system prompt", "hack"]
    for word in forbidden_keywords:
        if word in query_lower:
            print(f"[Guardrail Warning] Blocked potential injection attempt keyword: '{word}'")
            return False
   
    if len(user_query.strip()) < 3:
        print("[Guardrail Warning] Query too short.")
        return False

    return True

def output_guardrail(response_text: str) -> str:
    """Inspects and cleans the generated LLM response."""
    if not response_text or len(response_text.strip()) == 0:
        return "Guardrail Error: The model returned an empty response."
    
    # Example guardrail: Ensure no raw system tags leaked out
    cleaned_response = response_text.replace("<|im_start|>","").replace("<|im_end|>","")
    return cleaned_response

# ==========================================
# RAG PIPELINE SETUP
# ==========================================
def initialize_vector_db():
    """Loads documents, chunks them, and saves to local Chroma vector store."""
    os.makedirs(DATA_PATH, exist_ok=True)

    # Only create the sample file if the data folder is empty
    if not os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, "rag_sample.txt"), "w") as f:
            f.write("Ollama allows developers to run large language models locally on their hardware."
                    "ChromaDB is an open-source vector database for storing and querying embeddings.")

    print("Loading documents....")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("Generating local embeddings and saving to ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_PATH)
    return vector_store
        
def get_or_create_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        return initialize_vector_db()
    
# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("---Initializing Local RAG System---")
    check_ollama()
    vector_store = get_or_create_db()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Initialize Local Ollama LLM
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    
    # Prompt template ensuring grounding
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve -> format -> prompt -> llm -> parse
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n---Local RAG with Guardrails is Ready! (Type 'exit' to quit)---")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break;
        
        # 1. Apply Input Guardrail
        if not input_guardrail(user_input):
            print("Response [Blocked by Input Guardrail] Your query violated safety or formatting standards.")
            continue
        
        # 2. Execute RAG Retrieval & Generation
        try:
            raw_answer = rag_chain.invoke(user_input)
            
            # 3. Apply Output Guardrail
            final_answer = output_guardrail(raw_answer)
            
            print(f"\nAnswer: {final_answer}")
        except Exception as e:
            print("An error occurred during retrieval or generation:", str(e))
            
if __name__ == "__main__":
    main()