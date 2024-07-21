# models/transformer.py
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, time_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.time_embedding = nn.Linear(time_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, model_dim))
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, model_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, output_dim)
    
    def forward(self, x, time):
        input_embedded = self.input_embedding(x)
        time_embedded = self.time_embedding(time)
        x = input_embedded + time_embedded + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# Example parameters (tune as needed)
input_dim = 5  # Adjust this according to your input dimension
time_dim = 4   # Adjust this according to your time feature dimension
model_dim = 64
num_heads = 8
num_layers = 4
output_dim = 1

def get_model():
    return TimeSeriesTransformer(input_dim, time_dim, model_dim, num_heads, num_layers, output_dim)
