# filename: train_couplet.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the dataset class
class CoupletDataset(Dataset):
    def __init__(self, in_file, out_file, vocabs_file):
        with open(in_file, 'r', encoding='utf-8') as f:
            self.in_lines = f.readlines()
        with open(out_file, 'r', encoding='utf-8') as f:
            self.out_lines = f.readlines()
        with open(vocabs_file, 'r', encoding='utf-8') as f:
            self.vocabs = f.readlines()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self):
        return len(self.in_lines)

    def __getitem__(self, idx):
        in_text = self.in_lines[idx].strip()
        out_text = self.out_lines[idx].strip()
        in_tokens = self.tokenizer.encode_plus(in_text, add_special_tokens=True, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        out_tokens = self.tokenizer.encode_plus(out_text, add_special_tokens=True, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'input_ids': in_tokens['input_ids'].squeeze(),
            'attention_mask': in_tokens['attention_mask'].squeeze(),
            'labels': out_tokens['input_ids'].squeeze()
        }

# Define the model
class CoupletModel(nn.Module):
    def __init__(self, num_labels):
        super(CoupletModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

# Load the data and create the dataloaders
dataset = CoupletDataset(in_file='./couplet/in.txt', out_file='./couplet/out.txt', vocabs_file='./couplet/vocabs')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Create the model and optimizer
model = CoupletModel(len(dataset.vocabs)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
total_loss = 0
total_correct = 0
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs.loss
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=1)
        total_correct += torch.sum(predictions == labels)
accuracy = total_correct / len(dataset)
print(f'Evaluation results: Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'couplet_model.pth')