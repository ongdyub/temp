import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_length=2048):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.chord_embed = nn.Embedding(chord_size, d_model)
        self.inst_embed = nn.Embedding(inst_size, d_model)
        
        self.pos_encoder = nn.Parameter(torch.randn(max_length, d_model))
        
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        
        self.chord_proj = nn.Linear(d_model, chord_size)
        self.inst_proj = nn.Linear(d_model, inst_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.embed(src) + self.pos_encoder[:src.size(1)]
        tgt = self.embed(tgt) + self.pos_encoder[:tgt.size(1)]
        src = src * torch.sqrt(torch.tensor(self.d_model))
        tgt = tgt * torch.sqrt(torch.tensor(self.d_model))
        memory = self.transformer.encoder(src, mask=src_mask)
        outs = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.out(outs)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

# Example usage
vocab_size = 10000  # Define the vocabulary size
d_model = 512      # Embedding dimension
nhead = 8          # Number of heads in multi-head attention models
num_encoder_layers = 6  # Number of sub-encoder-layers in the encoder
num_decoder_layers = 6  # Number of sub-decoder-layers in the decoder
dim_feedforward = 2048  # Dimension of feedforward network

model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

# Example data
bsz = 10  # Batch size
src = torch.randint(0, vocab_size, (bsz, 2048))  # Randomly generated input
tgt = torch.randint(0, vocab_size, (bsz, 2048))  # Randomly generated target

# Masks
src_mask = generate_square_subsequent_mask(2048)
tgt_mask = generate_square_subsequent_mask(2048)

# Forward pass
output = model(src, tgt, src_mask, tgt_mask)
print(output.shape)  # Expected shape: [batch_size, max_length, vocab_size]
