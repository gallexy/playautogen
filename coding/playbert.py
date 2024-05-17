# filename: playbert.py
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Define the text with a masked word
text = "今天天气[MASK]好。"

# Tokenize the text
input_ids = tokenizer.encode(text, return_tensors='pt')

# Find the index of the masked word
masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].item()

# Predict the masked word
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

# Get the predicted token and convert it back to text
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)