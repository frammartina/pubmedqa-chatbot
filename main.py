import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
import os

MODEL_PATH = "LSTM__0.9170.pt.pt"
MODEL_URL = "https://drive.google.com/file/d/133F-sRp_mCGOo73t1ieSnbk5fSxPFENT/view?usp=drive_link"  

if not os.path.exists(MODEL_PATH):
    import gdown
    print("Scaricamento dei pesi dal Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=False)
model_name = "dmis-lab/biobert-base-cased-v1.1"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("LSTM__0.9170.pt.pt", map_location=device))
model.to(device)
model.eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    context: str
    long_answer: str

@app.post("/chat")
def get_response(query: Query):
    text = query.question + " " + query.context + " " + query.long_answer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    answer = torch.argmax(outputs.logits, dim=-1).item()
    result = "Yes" if answer == 1 else "No"
    return {"answer": result}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
