from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_app import get_answer_from_graph
from image import router as image_router  # ✅ Import the image router

app = FastAPI()

# ✅ Add CORS middleware to allow mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains/IPs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing RAG QA route
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Question):
    answer = get_answer_from_graph(data.question)
    return {"answer": answer}

# ✅ Register the image generation routes
app.include_router(image_router)
