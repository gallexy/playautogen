# filename: playbert.py
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load the pre-trained BERT model and tokenizer (for Chinese)
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Example sentence with a masked token
text = "今天[MASK]很好。"

# Encode the sentence
input_ids = tokenizer.encode(text, return_tensors="pt")

# Predict the masked token
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

# Decode the prediction and find the most likely token
predicted_token_id = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.decode(predicted_token_id)

# Print the predicted token
print("Predicted token:", predicted_token)