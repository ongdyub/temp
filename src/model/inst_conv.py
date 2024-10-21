import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import math
from transformers import ASTConfig, ASTModel, GPT2Config, GPT2Model, AutoModelForCausalLM, GPT2LMHeadModel, BartConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Broadcasting positional encodings
        return x  # Shape: (batch_size, seq_len, d_model)
    
class NormEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(NormEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Broadcasting positional encodings
        return x  # Shape: (batch_size, seq_len, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # d_model must be divisible by num_heads
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])  # Q, K, V, and output projections

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        # Linear projections
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # After projection and reshaping: (batch_size, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, value)
        # attn_output shape: (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](attn_output)  # Shape: (batch_size, seq_len, d_model)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.linear2(F.relu(self.linear1(x)))  # Shape: (batch_size, seq_len, d_model)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        attn_output = self.self_attn(x, x, x, mask)  # Self-attention
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm1(x)

        ff_output = self.feed_forward(x)  # Feed forward
        x = x + self.dropout(ff_output)  # Add & Norm
        return self.layernorm2(x)  # Shape: (batch_size, seq_len, d_model)
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # Shape: (batch_size, src_len, d_model)
    
    
import torch
import torch.nn as nn

class ConvNet2D(nn.Module):
    def __init__(self, d_model, output_dim=133):
        super(ConvNet2D, self).__init__()
        
        # Define 2D convolutions with padding to maintain (src_len, d_model) shape
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        
        self.relu = nn.ReLU()

        # Final projection to match the output_dim
        self.final_conv = nn.Conv1d(in_channels=d_model, out_channels=output_dim, kernel_size=1)
    
    def forward(self, x):
        # x shape: (batch_size, src_len, d_model)
        # Add a channel dimension: (batch_size, 1, src_len, d_model)
        x = x.unsqueeze(1)
        
        # Apply the convolutional layers
        x = self.relu(self.conv1(x))  # (batch_size, 32, src_len, d_model)
        x = self.relu(self.conv2(x))  # (batch_size, 64, src_len, d_model)
        x = self.relu(self.conv3(x))  # (batch_size, 128, src_len, d_model)
        x = self.relu(self.conv4(x))  # (batch_size, 256, src_len, d_model)
        x = self.conv5(x)             # (batch_size, 1, src_len, d_model)

        # Squeeze to remove the extra channel dimension: (batch_size, src_len, d_model)
        x = x.squeeze(1)
        
        # Apply a 1D convolution to reduce the feature dimension to 133
        x = self.final_conv(x.transpose(1, 2))  # Transpose to (batch_size, d_model, src_len)
        x = x.transpose(1, 2)  # Back to (batch_size, src_len, 133)

        return x

# Example usage:
# batch_size = 32
# src_len = 50
# d_model = 64

# model = ConvNet2D(d_model=d_model)
# input_tensor = torch.randn(batch_size, src_len, d_model)
# output = model(input_tensor)
# print(output.shape)  # Expected shape: (batch_size, src_len, 133)

    
class InstTFEncoderConvolution(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=3, vocab_size=150, max_len=768, dropout=0.1):
        super(InstTFEncoderConvolution, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        self.conv_out = ConvNet2D(d_model=d_model)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
            
        x = self.conv_out(x) # Shape: (batch_size, src_len, d_model)
        
        return x  # Shape: (batch_size, src_len, 133)
    
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)