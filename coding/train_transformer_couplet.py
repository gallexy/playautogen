# filename: train_transformer_couplet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import os


class CoupletDataset(Dataset):
    def __init__(self, input_file, output_file, vocab):
        self.inputs = open(input_file, 'r', encoding='utf-8').readlines()
        self.outputs = open(output_file, 'r', encoding='utf-8').readlines()
        self.src_vocab = {word.strip(): i for i, word in enumerate(open(vocab, 'r', encoding='utf-8').readlines())}

        # Ensure <unk> token is in the vocabulary
        if "<unk>" not in self.src_vocab:
            self.src_vocab["<unk>"] = len(self.src_vocab)
        self.tgt_vocab = self.src_vocab
        self.src_inv_vocab = {i: word for word, i in self.src_vocab.items()}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_seq = self.inputs[idx].strip().split()
        tgt_seq = self.outputs[idx].strip().split()

        src_indices = [self.src_vocab.get(word, self.src_vocab["<unk>"]) for word in src_seq]
        tgt_indices = [self.tgt_vocab.get(word, self.tgt_vocab["<unk>"]) for word in tgt_seq]

        return torch.tensor(src_indices), torch.tensor(tgt_indices), idx


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        self.generator = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.src_tok_emb(src) * math.sqrt(self.d_model)
        tgt = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        output = self.transformer(src, tgt)
        output = self.generator(output)
        return output

    def generate(self, src, max_len=20):
        memory = self.transformer.encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        ys = torch.ones(1, 1).fill_(0).type_as(src.data)
        for i in range(max_len - 1):
            tgt = self.tgt_tok_emb(ys) * math.sqrt(self.d_model)
            out = self.transformer.decoder(tgt, memory)
            prob = self.generator(out[:, -1])
            next_word = torch.argmax(prob, dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)
            if next_word == 3:  # Assuming 3 is the EOS token index
                break
        return ys


def collate_fn(batch):
    src_seqs, tgt_seqs, idxs = zip(*batch)
    src_seqs = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_seqs = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_seqs, tgt_seqs


def train():
    dataset = CoupletDataset('./couplet/in.txt', './couplet/out.txt', './couplet/vocabs')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = TransformerModel(vocab_size=len(dataset.src_vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Reduce the number of epochs for memory management
        model.train()
        total_loss = 0
        for i, (src_seqs, tgt_seqs) in enumerate(dataloader):
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)

            tgt_input = tgt_seqs[:, :-1]
            tgt_output = tgt_seqs[:, 1:]

            optimizer.zero_grad()
            output = model(src_seqs, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

    # Save the model
    torch.save(model.state_dict(), 'transformer_couplet.pth')


def evaluate():
    dataset = CoupletDataset('./couplet/in.txt', './couplet/out.txt', './couplet/vocabs')
    model = TransformerModel(vocab_size=len(dataset.src_vocab))
    model.load_state_dict(torch.load('transformer_couplet.pth'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Evaluate on a few sample inputs
    samples = ["上联示例1", "上联示例2", "上联示例3"]  # Replace with actual samples from your data
    for sample in samples:
        src_seq = [dataset.src_vocab.get(word, dataset.src_vocab["<unk>"]) for word in sample.split()]
        src_tensor = torch.tensor(src_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_seq = model.generate(src_tensor)
        predicted_words = [dataset.src_inv_vocab[idx.item()] for idx in predicted_seq.squeeze()]
        print(f"Input: {sample}")
        print(f"Output: {' '.join(predicted_words)}")


if __name__ == '__main__':
    train()
    evaluate()