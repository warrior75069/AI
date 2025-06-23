from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Load Phi-2 model
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create a pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Request model
class QueryRequest(BaseModel):
    question: str
    max_length: int = 150

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        prompt = f"Q: {request.question}\nA:"
        result = qa_pipeline(prompt, max_new_tokens=request.max_length)
        answer = result[0]["generated_text"].replace(prompt, "").strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
