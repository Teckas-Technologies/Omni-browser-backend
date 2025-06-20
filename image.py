import os
import openai
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ImageRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    aspect_ratio: Optional[str] = None  # e.g., "1:1", "16:9", "9:16"

@router.post("/generate-image")
def generate_image(data: ImageRequest):
    # Handle optional style
    if data.style:
        formatted_prompt = f"{data.prompt} in {data.style} style"
    else:
        formatted_prompt = data.prompt

    # Handle optional aspect_ratio with default fallback
    size_map = {
        "1:1": "1024x1024",
        "16:9": "1792x1024",
        "9:16": "1024x1792",
    }
    image_size = size_map.get(data.aspect_ratio, "1024x1024")

    response = openai.images.generate(
        model="dall-e-3",
        prompt=formatted_prompt,
        size=image_size,
        quality="standard",
        n=1,
    )

    return {"image_url": response.data[0].url}
