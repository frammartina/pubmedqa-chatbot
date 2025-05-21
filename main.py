from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyngrok import ngrok
from google.colab import userdata
from uvicorn import Config, Server



ngrok_token = os.environ["NGROK_AUTH_TOKEN"]

ngrok.set_auth_token(ngrok_token)

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=False)
model_name = "dmis-lab/biobert-base-cased-v1.1"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Binary classification
)


# Adjust the map_location based on your device ('cuda' if available, 'cpu' otherwise)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("LSTM__0.9170.pt.pt", map_location=device))
model.to(device)
model.eval()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione specifica il dominio della tua pagina web
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
    # Preprocess input
    text = query.question + " " + query.context + " " + query.long_answer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    answer = torch.argmax(outputs.logits, dim=-1).item()
    result = "Yes" if answer == 1 else "No"
    return {"answer": result}

# Expose the app through ngrok
public_url = ngrok.connect(8000).public_url
print(f"FastAPI is running on {public_url}")
# Run FastAPI app with uvicorn
config = Config(app=app, host="0.0.0.0", port=8000)
server = Server(config)
import nest_asyncio
nest_asyncio.apply()  # Necessario per evitare problemi con gli eventi in Colab
await server.serve()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
