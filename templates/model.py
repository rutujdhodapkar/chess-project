import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import math
import os
import chess
import random

# ---------------------- Device Setup ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------- Dataset & Vocabulary Setup ----------------------
# Use a relative file path for cloud deployable environments
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'processed_dataset.csv')
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Warning: '{dataset_path}' not found. Using empty dataset.")
    df = pd.DataFrame(columns=['player1_moves', 'player2_moves'])

def tokenize(moves_str):
    if pd.isna(moves_str):
        return []
    return moves_str.strip().split()

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
all_moves = []
for _, row in df.iterrows():
    white_tokens = tokenize(row.get('player1_moves', ''))
    black_tokens = tokenize(row.get('player2_moves', ''))
    all_moves.extend(white_tokens)
    all_moves.extend(black_tokens)
move_counts = Counter(all_moves)
for token in move_counts:
    if token not in vocab:
        vocab[token] = len(vocab)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)
inv_vocab = {idx: token for token, idx in vocab.items()}

def encode_moves(moves_list, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in moves_list]

max_seq_length = 0
encoded_white_moves = []
encoded_black_moves = []
for _, row in df.iterrows():
    white_tokens = tokenize(row.get('player1_moves', ''))
    max_seq_length = max(max_seq_length, len(white_tokens))
    encoded_white_moves.append(encode_moves(white_tokens, vocab))
    
    black_tokens = tokenize(row.get('player2_moves', ''))
    if len(black_tokens) == 0:
        encoded_black_moves.append([vocab[PAD_TOKEN]])
    else:
        encoded_black_moves.append(encode_moves(black_tokens, vocab)[0:1])

def pad_sequence(seq, max_len, pad_value):
    return seq + [pad_value] * (max_len - len(seq))

padded_white_moves = [pad_sequence(seq, max_seq_length, vocab[PAD_TOKEN]) for seq in encoded_white_moves]
white_data = np.array(padded_white_moves)
black_data = np.array([seq[0] if isinstance(seq, list) else seq for seq in encoded_black_moves])

train_white, test_white, train_black, test_black = train_test_split(
    white_data, black_data, test_size=0.2, random_state=42
)
class ChessDataset(Dataset):
    def __init__(self, white_moves, black_moves):
        self.white_moves = white_moves
        self.black_move = black_moves
        
    def __len__(self):
        return len(self.white_moves)
    
    def __getitem__(self, idx):
        return {
            'white_moves': torch.tensor(self.white_moves[idx], dtype=torch.long),
            'black_move': torch.tensor(self.black_move[idx], dtype=torch.long)
        }
train_dataset = ChessDataset(train_white, train_black)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------------------- Define the Neural Network ----------------------
class MovePredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=2, bidirectional=True, target_params=3000000):
        super(MovePredictor, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        hidden_dim = 64
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD_TOKEN])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.3)
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, vocab_size)
        
        self.adv_fc = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)
        )
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(embeds)
        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_dim)
            final_hidden = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=1)
        else:
            final_hidden = h_n[-1]
        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)
        output = self.adv_fc(output)
        return output

model_net = MovePredictor(vocab_size).to(device)
total_params = sum(p.numel() for p in model_net.parameters())
print("Model total parameters:", total_params)

# Use a relative model path for cloud deployment
model_path = os.path.join(os.path.dirname(__file__), 'models', 'move_predictor.pth')
if os.path.exists(model_path):
    model_net.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded saved model from", model_path)
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_net.parameters(), lr=0.001)
    num_epochs = 3
    model_net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            white_moves = batch['white_moves'].to(device)
            targets = batch['black_move'].to(device).squeeze()
            optimizer.zero_grad()
            outputs = model_net(white_moves)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * white_moves.size(0)
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    torch.save(model_net.state_dict(), model_path)
    print("Model saved to", model_path)

# ---------------------- Exposed Functions ----------------------
def store_move(move):
    """
    Log the move to a file.
    """
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'moves_log.txt')
    try:
        with open(log_path, "a") as f:
            f.write(move + "\n")
    except Exception as e:
        print(f"Error writing move to log: {e}")

def predict_moves(move_history):
    """
    Given a list of move strings (SAN), use the ML model to predict the next move.
    Now only the most recent moves (up to max_seq_length) are used.
    """
    tokens = []
    for move in move_history:
        tokens.extend(tokenize(move))
    if not tokens:
        return []
    # Only consider the most recent tokens (truncating if necessary)
    encoded = encode_moves(tokens, vocab)
    if len(encoded) > max_seq_length:
        encoded = encoded[-max_seq_length:]
    padded = pad_sequence(encoded, max_seq_length, vocab[PAD_TOKEN])
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
    model_net.eval()
    with torch.no_grad():
        output = model_net(input_tensor)  # (1, vocab_size)
        topk = torch.topk(output, 3)
        indices = topk.indices[0].cpu().numpy()
    predictions = []
    for idx in indices:
        token = inv_vocab.get(idx, UNK_TOKEN)
        predictions.append(token)
    return predictions

if __name__ == '__main__':
    sample_history = ["e4"]
    print("Predicted next moves:", predict_moves(sample_history))
