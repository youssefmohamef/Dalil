# main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
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

# ============================================================
# PATHS
# ============================================================
BASE_PATH = os.getcwd()
SOURCE_DATA_PATH = os.path.join(BASE_PATH, "Source_data")
VECTOR_DB_PATH   = os.path.join(BASE_PATH, "vector_db_index")
LOG_FILE_PATH    = os.path.join(BASE_PATH, "admin_knowledge_base.txt")

# ============================================================
# LLM & EMBEDDINGS
# ============================================================
print("⏳ Loading LLM and embeddings...")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
chat_history = []

def chat(question):
    global chat_history
    result = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    answer = result["answer"]
    chat_history.append({"user": question})
    chat_history.append({"assistant": answer})
    return answer

# ============================================================
# ADMIN ENGINE
# ============================================================
def internal_selective_delete(name, category=""):
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

def dalil_main_engine(user_input, is_admin=False):
    global vectorstore
    if is_admin:
        if user_input.lower().startswith("delete:"):
            target = user_input.replace("delete:", "").strip()
            internal_selective_delete(target, "")
            return f"🗑️ Records for '{target}' removed from overrides."
        elif user_input.lower().startswith("insert:"):
            try:
                content = user_input.replace("insert:", "").strip()
                parts   = [p.strip() for p in content.split(",")]
                if len(parts) >= 3:
                    name, category, value = parts[0], parts[1], parts[2]
                    new_record = f"UPDATED INFO: {name}'s {category} is {value}"
                    internal_selective_delete(name, category)
                    vectorstore.add_documents([Document(page_content=new_record)])
                    vectorstore.save_local(VECTOR_DB_PATH)
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(new_record + "\n")
                    return f"✅ Successfully Inserted: {name}'s {category} record."
                return "⚠️ Format Error! Use: insert: Name, Category, Value"
            except Exception as e:
                return f"❌ System Error: {str(e)}"
    docs    = vectorstore.similarity_search(user_input, k=5)
    context = "\n".join([d.page_content for d in docs])
    system_instruction = (
        "You are an expert assistant. Use the provided context to answer. "
        "IMPORTANT: If you find conflicting information, prioritize records labeled 'UPDATED INFO'."
    )
    final_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {user_input}"
    return llm.invoke(final_prompt).content

# ============================================================
# AUTH SETUP
# ============================================================
print("⏳ Setting up auth...")
SECRET_KEY           = "dalil-super-secret-key-change-in-production"
ALGORITHM            = "HS256"
TOKEN_EXPIRE_MINUTES = 60

pwd_context   = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

print("⏳ Hashing passwords...")
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": pwd_context.hash("admin123"),
        "role": "admin"
    },
    "student": {
        "username": "student",
        "password": pwd_context.hash("student123"),
        "role": "user"
    }
}
print("✅ Auth ready.")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role     = payload.get("role")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================
# FASTAPI APP
# ============================================================
print("⏳ Starting FastAPI app...")
app = FastAPI(title="Dalil University Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST MODELS
# ============================================================
class AskRequest(BaseModel):
    question: str

class AdminRequest(BaseModel):
    command: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"status": "✅ Dalil API is running"}

@app.post("/register")
def register(req: RegisterRequest):
    if req.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    if req.role not in ["user", "admin"]:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
    USERS_DB[req.username] = {
        "username": req.username,
        "password": pwd_context.hash(req.password),
        "role": req.role
    }
    return {
        "message": f"✅ User '{req.username}' registered successfully",
        "role": req.role
    }

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Wrong username or password")
    token = create_token({"sub": user["username"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"]
    }

@app.post("/ask")
def ask(req: AskRequest, current_user: dict = Depends(get_current_user)):
    answer = chat(req.question)
    return {
        "answer": answer,
        "asked_by": current_user["username"],
        "role": current_user["role"]
    }

@app.post("/admin/command")
def admin_command(req: AdminRequest, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="❌ Admins only. Access denied.")
    result = dalil_main_engine(req.command, is_admin=True)
    return {
        "result": result,
        "executed_by": current_user["username"]
    }

@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "role": current_user["role"]
    }

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)