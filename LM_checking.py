# We will use a small LM to check if OCR'd sentences have mistakes

from transformers import BertTokenizer, BertForMaskedLM
import torch

def load_model(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def score_sentence(sentence, tokenizer, model):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, labels=tensor_input).loss
    return -loss.item() / len(tokenize_input)  # Average log-likelihood per word

# Load the model
tokenizer, model = load_model("chinese-macbert-base")
