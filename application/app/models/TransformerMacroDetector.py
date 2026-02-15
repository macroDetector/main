import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class TransformerMacroAutoencoder(nn.Module):
    def __init__(self, input_size=5, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        # 입력 피처를 d_model 차원으로 투영
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))  # 최대 seq_len
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder: 다시 input_size 차원으로 복원
        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        seq_len = x.size(1)
        x_emb = self.embedding(x)
        x_emb = x_emb + self.pos_encoder[:, :seq_len, :]
        
        x_encoded = self.transformer_encoder(x_emb)
        # 전체 시퀀스 재구성
        x_recon = self.decoder(x_encoded)
        
        return x_recon

class MacroDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]