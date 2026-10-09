import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, hidden_size=None, n_layer=1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if hidden_size is None:
            hidden_size = d_model
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=n_layer, batch_first=True)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, idx):
        x = self.token_emb(idx)
        x, _ = self.lstm(x)
        return self.head(self.ln_f(x))
