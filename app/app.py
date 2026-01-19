from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from model import load_model

# Define API input
class UtteranceRequest(BaseModel):
    utterance: str

# Load model + tokenizer
intent2id = {"check_application_status": 0, "start_new_application": 1, "reset_password_login_help": 2,
             "service_eligibility": 3, "requirements_information": 4}  # Example mapping
id2intent = {v: k for k, v in intent2id.items()}

model, tokenizer = load_model("xlmr_intent_classifier.pt", intent2id)

app = FastAPI(title="Voice AI Intent Classification API")

CONFIDENCE_THRESHOLD = 0.5  # fallback threshold

@app.post("/predict")
def predict_intent(request: UtteranceRequest):
    utterance = request.utterance
    if not utterance:
        raise HTTPException(status_code=400, detail="Utterance cannot be empty")

    # Tokenize
    inputs = tokenizer(
        utterance,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(model.encoder.device)
    attention_mask = inputs["attention_mask"].to(model.encoder.device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    intent = id2intent[int(pred_idx)]
    confidence = float(conf)

    # Handle low-confidence predictions
    if confidence < CONFIDENCE_THRESHOLD:
        intent = "fallback"
        confidence = float(conf)  # still report the highest prob

    return {
        "utterance": utterance,
        "predicted_intent": intent,
        "confidence": confidence
    }
