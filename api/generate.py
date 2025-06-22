from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.responses import JSONResponse
from mangum import Mangum

app = FastAPI()

# Load the lightweight model once
generator = pipeline('text-generation', model='distilgpt2')

class ProductDescriptionRequest(BaseModel):
    prompt: str
    max_length: int = 450

@app.post("/generate")
async def generate_description(request: ProductDescriptionRequest):
    try:
        result = generator(request.prompt, max_length=request.max_length, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        return {"description": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Required for Vercel
handler = Mangum(app)
