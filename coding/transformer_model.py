# filename: transformer_model.py
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ro")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ro")

print(tokenizer)
print(model)

def translate(text, src_lang="en", tgt_lang="ro"):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output = model.generate(input_ids=input_ids)
    return tokenizer.batch_decode(output, skip_special_tokens=True)

text = "Hello, world!"
print(translate(text))