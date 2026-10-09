import torch
import torch.nn as nn
from model_transformer import BaselineTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=1, batch_first=True)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, idx):
        x = self.token_emb(idx)
        x, _ = self.lstm(x)
        return self.head(self.ln_f(x))

vocab_size = 21 # from Vocab
d_model = 256
n_head = 4
n_layer = 1
max_len = 320
pos = 'learned'

transformer = BaselineTransformer(vocab_size, d_model, n_head, n_layer, pos, max_len)
target_params = count_parameters(transformer)

print(f"Transformer params: {target_params}")

# find hidden_size that matches target_params
best_h = 1
min_diff = float('inf')
for h in range(1, 2048):
    lstm = LSTMModel(vocab_size, d_model, h)
    params = count_parameters(lstm)
    diff = abs(params - target_params)
    if diff < min_diff:
        min_diff = diff
        best_h = h

lstm = LSTMModel(vocab_size, d_model, best_h)
print(f"Best LSTM hidden_size: {best_h}, params: {count_parameters(lstm)}, diff: {min_diff}")

