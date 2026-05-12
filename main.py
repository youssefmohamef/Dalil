# IMPORTS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# ============================================================
# PATHS
# ============================================================
BASE_PATH = os.getcwd()
SOURCE_DATA_PATH = os.path.join(BASE_PATH, "data", "Source_data")
VECTOR_DB_PATH   = os.path.join(BASE_PATH, "vector_db_index")
LOG_FILE_PATH    = os.path.join(BASE_PATH, "admin_knowledge_base.txt")

# ============================================================
# LLM & EMBEDDINGS
# ============================================================

print("⏳ Loading LLM and embeddings...")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("✅ LLM and embeddings ready.")

# ============================================================
# VECTORSTORE
# ============================================================
def get_or_create_vectorstore(embedding_model):
    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        print("✅ Vector database found. Loading...")
        try:
            vs = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
            print("✅ Vector database loaded successfully.")
            return vs
        except Exception as e:
            print(f"⚠️ Failed to load existing database: {e}. Rebuilding...")
            import shutil
            shutil.rmtree(VECTOR_DB_PATH)
    print("🆕 Building vector database...")
    all_docs = []   
    if os.path.exists(SOURCE_DATA_PATH):
        all_docs.extend(DirectoryLoader(SOURCE_DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader).load())
        all_docs.extend(DirectoryLoader(SOURCE_DATA_PATH, glob="*.csv", loader_cls=CSVLoader, loader_kwargs={"encoding": "utf-8"}).load())
        all_docs.extend(DirectoryLoader(SOURCE_DATA_PATH, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load())
    if not all_docs:
        return FAISS.from_documents([Document(page_content="Initial System")], embedding_model)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(all_docs)
    vs = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vs.save_local(VECTOR_DB_PATH)
    return vs

vectorstore = get_or_create_vectorstore(embedding_model)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})

# ============================================================
# PROMPT & RAG CHAIN
# ============================================================
system_prompt_text = """You are an expert university teaching assistant.

Use the retrieved context ONLY as reference.
DO NOT copy the text directly.
DO NOT list chunks.

Your task:
1. Understand the user's question.
2. Extract relevant ideas from the context.
3. Reason step by step.
4. Generate a clear, original answer in your own words.

If the context is insufficient, say so.

ALWAYS follow this format:
**Answer:**
<direct answer>

**Details:**
- <bullet point 1>
- <bullet point 2>

Context:
{context}

Question:
{input}

Answer:
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    ("human", "{input}")
])

print("⏳ Building RAG chain...")
document_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
rag_chain      = create_retrieval_chain(retriever, document_chain)
print("✅ RAG chain ready.")

# ============================================================
# CHAT FUNCTION
# ============================================================
def chat(question):

    docs = retriever.get_relevant_documents(
        question
    )

    # ============================================
    # detect deleted entities
    # ============================================
    deleted_entities = []

    for d in docs:

        content = d.page_content.lower()

        if content.startswith("deleted info:"):

            deleted_name = (
                content
                .replace("deleted info:", "")
                .strip()
                .split(",")[0]
            )

            deleted_entities.append(
                deleted_name
            )

    # ============================================
    # filter deleted docs
    # ============================================
    filtered_docs = []

    for d in docs:

        content_lower = d.page_content.lower()

        is_deleted_related = False

        for deleted_name in deleted_entities:

            if (
                deleted_name in content_lower
                and not content_lower.startswith("deleted info:")
            ):

                is_deleted_related = True
                break

        if not is_deleted_related:

            filtered_docs.append(d)

    # ============================================
    # if no valid docs
    # ============================================
    if not filtered_docs:

        return (
            "I could not find the answer "
            "in the database."
        )

    # ============================================
    # invoke chain manually
    # ============================================
    result = document_chain.invoke({

        "context": filtered_docs,

        "input": question

    })

    return result

# ============================================================
# ADMIN ENGINE
# ============================================================
def save_admin_log(name, category=""):
    global vectorstore
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = [
            l for l in lines
            if not (name.lower().strip() in l.lower() and category.lower().strip() in l.lower())
        ]
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    return False

def dalil_main_engine(user_input):

    global vectorstore

    # INSERT
    # ========================================================
    if user_input.lower().startswith("insert:"):

        try:

            content = user_input.replace(
                "insert:","").strip()

            parts = [
                p.strip()
                for p in content.split(",")
            ]

            if len(parts) < 3:

                return (
                    "❌ Use format:\n"
                    "insert: name, category, value"
                )

            name = parts[0]
            category = parts[1]
            value = parts[2]

            new_record = (
                f"UPDATED INFO: "
                f"{name}'s {category} is {value}"
            )

            vectorstore.add_documents([
                Document(
                    page_content=new_record
                )
            ])

            vectorstore.save_local(
                VECTOR_DB_PATH
            )

            save_admin_log(new_record)

            return (
                f"✅ Inserted successfully:\n"
                f"{new_record}"
            )

        except Exception as e:

            return f"❌ Error: {str(e)}"

    # UPDATE
    # ========================================================
    elif user_input.lower().startswith("update:"):

        try:

            content = user_input.replace(
                "update:","").strip()

            parts = [
                p.strip()
                for p in content.split(",")
            ]

            if len(parts) < 3:

                return (
                    "❌ Use format:\n"
                    "update: name, category, new_value"
                )

            name = parts[0]
            category = parts[1]
            new_value = parts[2]

            updated_record = (
                f"UPDATED INFO: "
                f"{name}'s {category} is {new_value}"
            )

            vectorstore.add_documents([
                Document(
                    page_content=updated_record
                )
            ])

            vectorstore.save_local(
                VECTOR_DB_PATH
            )

            save_admin_log(updated_record)

            return (
                f"✅ Updated successfully:\n"
                f"{updated_record}"
            )

        except Exception as e:

            return f"❌ Error: {str(e)}"

    # DELETE
    # ========================================================
    elif user_input.lower().startswith("delete:"):

        try:

            target = user_input.replace(
                "delete:","").strip()

            delete_record = (f"DELETED INFO: {target}")

            vectorstore.add_documents([
                Document(
                    page_content=delete_record
                )
            ])

            vectorstore.save_local(
                VECTOR_DB_PATH
            )

            save_admin_log(delete_record)

            return (
                f"✅ Delete marker added for:\n"
                f"{target}"
            )

        except Exception as e:

            return f"❌ Error: {str(e)}"

    return "❌ Unknown command."

# ============================================================
# # FASTAPI APP INITIALIZATION
# ============================================================
app = FastAPI(title="Dalil University Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API MODELS
# ============================================================
class AskRequest(BaseModel):
    """Schema for AI chatbot queries."""
    question: str

class AdminRequest(BaseModel):
    """Schema for administrative database commands."""
    command: str

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "✅ Dalil API is running", "timestamp": datetime.utcnow()}

@app.post("/ask")
def ask(req: AskRequest):

    forbidden_commands = [
        "insert:",
        "update:",
        "delete:"
    ]

    question_lower = req.question.lower()

    for cmd in forbidden_commands:

        if question_lower.startswith(cmd):

            raise HTTPException(
                status_code=403,
                detail=("Admin commands are forbidden"
                        "❌ Admin commands "
            )
            )
    answer = chat(req.question)

    return {
        "answer": answer
    }
# ADMIN COMMAND ENDPOINT
# ============================================================
@app.post("/admin/command")
def admin_command(req: AdminRequest):

    result = dalil_main_engine(
        req.command
    )

    return {
        "result": result
    }

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)