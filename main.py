from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_app import get_answer_from_graph
from image import router as image_router  # ✅ Import the image router

app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Chat history store (in-memory list)
chat_history = []

# ✅ POST /ask endpoint
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Question):
    answer = get_answer_from_graph(data.question)
    # Save the exchange in history
    chat_history.append({"question": data.question, "answer": answer})
    return {"answer": answer}

# ✅ GET /history endpoint
@app.get("/history")
def get_chat_history():
    return {"history": chat_history}

# ✅ Register image generation routes
app.include_router(image_router)
