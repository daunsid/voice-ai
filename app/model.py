import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer

class XLMRIntentClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_labels=10):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # <s> token
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)
        return logits

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, intent2id):
    num_labels = len(intent2id)
    model = XLMRIntentClassifier(num_labels=num_labels)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint["config"]["model_name"])
    return model, tokenizer
