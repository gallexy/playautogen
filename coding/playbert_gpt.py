# filename: playbert_gpt.py
from transformers import BertTokenizer, BertModel
import torch

def test_bert_base_chinese():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # Encode text
    input_text = "中文文本处理"
    encoded_input = tokenizer(input_text, return_tensors='pt')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-chinese')

    # Predict hidden states features for each layer
    with torch.no_grad():
        output = model(**encoded_input)

    # Just return the hidden states of the last layer
    last_hidden_states = output.last_hidden_state

    print("Shape of last hidden states: ", last_hidden_states.shape)

if __name__ == "__main__":
    test_bert_base_chinese()