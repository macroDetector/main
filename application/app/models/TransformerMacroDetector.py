import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class TransformerMacroAutoencoder(nn.Module):
    def __init__(self,input_size: int,d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, bottleneck_ratio: int = 8):
        super().__init__()

        latent_dim = max(4, d_model // bottleneck_ratio) 

        # 입력 투영
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.latent_dropout = nn.Dropout(0.2)

        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)

        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, x, add_latent_noise=False, latent_noise_std=0.0):
        seq_len = x.size(1)
        x_emb = self.embedding(x)
        x_emb = x_emb + self.pos_encoder[:, :seq_len, :]

        encoded = self.transformer_encoder(x_emb)

        latent = self.to_latent(encoded)
        latent = self.latent_dropout(latent)

        if add_latent_noise:
            latent = latent + torch.randn_like(latent) * latent_noise_std

        decoded = self.from_latent(latent)
        x_recon = self.decoder(decoded)
        return x_recon



class MacroDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]