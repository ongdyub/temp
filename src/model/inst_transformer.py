import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import math
from transformers import ASTConfig, ASTModel, GPT2Config, GPT2Model, AutoModelForCausalLM, GPT2LMHeadModel, BartConfig

# class GPT2Model(nn.Module):
#     def __init__(self, vocab_size=140, n_embd=768, n_layer=12, n_head=12):
#         super(GPT2Model, self).__init__()
#         self.configuration = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, bos_token_id=2, eos_token_id=1)
#         self.model = GPT2LMHeadModel(self.configuration)
        
#     def get_embed(self, idx):
#         embedding_layer = self.model.transformer.wte
#         token_embedding = embedding_layer(torch.tensor([idx]))
#         return token_embedding
    
#     def extract_vocab_embeddings(self):
#         # Extract all the embeddings for the entire vocabulary
#         embedding_layer = self.model.transformer.wte
#         vocab_embeddings = embedding_layer.weight.detach().clone()
#         return vocab_embeddings

#     def forward(self, input_ids, labels=None, return_hidden_states=False):
#         attention_mask = self.make_mask(input_ids)
#         # Forward pass through the transformer to get hidden states
#         transformer_outputs = self.model.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

#         # Extract hidden states before the projection
#         hidden_states = transformer_outputs.last_hidden_state
        
#         if return_hidden_states:
#             return hidden_states

#         # Project the hidden states to vocabulary size
#         logits = self.model.lm_head(hidden_states)

#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.configuration.vocab_size), labels.view(-1))
#             return loss, logits
#         return logits

#     def make_mask(self, input_ids):
#         attention_mask = (input_ids != 0).long()
#         return attention_mask
    
#     def infer(self, input_ids, length=2048):
#         if len(input_ids.shape) == 1:
#             input_ids = input_ids.unsqueeze(0)
#         if len(input_ids.shape) > 2:
#             raise Exception
        
#         if length > 2048:
#             print("Max Length is 2048. Change Length Auto to 2048")
#             length = 2048
        
#         with torch.no_grad():
#             for step in range(length):
#                 logits = self.forward(input_ids)
#                 output = torch.argmax(logits, dim=2)

#                 predict = output[:,-1].unsqueeze(1)
#                 output_ids = torch.cat((input_ids, predict), dim=-1)

#                 input_ids = output_ids
                
#                 if output_ids.shape[1] > 2048:
#                     break

#         return output_ids

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
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # x shape: (batch_size, tgt_len, d_model)
        # memory shape: (batch_size, src_len, d_model)

        attn_output = self.self_attn(x, x, x, tgt_mask)  # Self-attention
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm1(x)

        attn_output = self.cross_attn(x, memory, memory, src_mask)  # Cross-attention
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm2(x)

        ff_output = self.feed_forward(x)  # Feed forward
        x = x + self.dropout(ff_output)  # Add & Norm
        return self.layernorm3(x)  # Shape: (batch_size, tgt_len, d_model)
    
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
    
class C2IEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(C2IEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.num_classes = 133
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 133)
        self.len_embed = nn.Embedding(770, d_model)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # Shape: (batch_size, src_len, d_model)
    
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)
    
    def proj_inst(self, x, length=None):
        # x shape: (batch_size, src_len, d_model)
        
        # Use the embedding corresponding to the BOS token (assumed to be the first token)
        bos_embedding = x[:, 0, :]  # Shape: (batch_size, d_model)
        
        if length == None:
            # First fully connected layer
            x = self.fc1(bos_embedding)  # Shape: (batch_size, d_model // 2)
            x = torch.relu(x)  # Apply ReLU activation
            
            # Second fully connected layer
            out = self.fc2(x)  # Shape: (batch_size, num_classes)
        else:
            # First fully connected layer
            len_embed = self.len_embed(length)
            len_embed = len_embed.squeeze(1)
            len_embed = len_embed/4

            bos_embedding = bos_embedding + len_embed
            x = self.fc1(bos_embedding)  # Shape: (batch_size, d_model // 2)
            x = torch.relu(x)  # Apply ReLU activation
            
            # Second fully connected layer
            out = self.fc2(x)  # Shape: (batch_size, num_classes)
        
        return out
    
    def avg_inst(self, x, length=None):
        # x shape: (batch_size, src_len, d_model)
        
        # Use the embedding corresponding to the BOS token (assumed to be the first token)
        bos_embedding = torch.mean(x, dim=1)  # Shape: (batch_size, d_model)
        
        if length == None:
            # First fully connected layer
            x = self.fc1(bos_embedding)  # Shape: (batch_size, d_model // 2)
            x = torch.relu(x)  # Apply ReLU activation
            
            # Second fully connected layer
            out = self.fc2(x)  # Shape: (batch_size, num_classes)
        else:
            # First fully connected layer
            len_embed = self.len_embed(length)
            len_embed = len_embed.squeeze(1)
            len_embed = len_embed/4
            bos_embedding = bos_embedding + len_embed
            x = self.fc1(bos_embedding)  # Shape: (batch_size, d_model // 2)
            x = torch.relu(x)  # Apply ReLU activation
            
            # Second fully connected layer
            out = self.fc2(x)  # Shape: (batch_size, num_classes)
        
        return out

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt shape: (batch_size, tgt_len)
        x = self.embedding(tgt)  # Embedding
        # x shape: (batch_size, tgt_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x  # Shape: (batch_size, tgt_len, d_model)
    
class InstDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super(InstDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        
        x = self.positional_encoding(tgt)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x  # Shape: (batch_size, tgt_len, d_model)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len, dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def generate_tgt_mask(self, tgt):
        # tgt shape: (batch_size, tgt_len)
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # Padding mask: (batch_size, 1, 1, tgt_len)
        nopeak_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()  # Look-ahead mask
        tgt_mask = tgt_mask & nopeak_mask.unsqueeze(0)  # Combined mask: (batch_size, 1, tgt_len, tgt_len)
        return tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src shape: (batch_size, src_len)
        # tgt shape: (batch_size, tgt_len)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        tgt_mask = self.generate_tgt_mask(tgt)  # Generate target mask

        memory = self.encoder(src, src_mask)  # Encoder output
        # memory shape: (batch_size, src_len, d_model)

        output = self.decoder(tgt, memory, src_mask, tgt_mask)  # Decoder output
        # output shape: (batch_size, tgt_len, d_model)

        return self.out(output)  # Final output projection, shape: (batch_size, tgt_len, tgt_vocab_size)
    
    def infer(self, src, x, length=766):
        with torch.no_grad():
            for step in range(length):
                output = self.forward(src, x)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((x, predict), dim=-1)

                x = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
        
    
    
class InstEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, inst_size, max_len, dropout=0.1):
        super(InstEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.inst_proj = nn.Linear(d_model, inst_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.inst_proj(x)
        return x  # Shape: (batch_size, src_len, inst_size)
    
class InstNoPEEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, inst_size, max_len, dropout=0.1):
        super(InstNoPEEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.inst_proj = nn.Linear(d_model, inst_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        # x shape: (batch_size, src_len, d_model)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        # x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.inst_proj(x)
        return x  # Shape: (batch_size, src_len, inst_size)

class InstNormEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, inst_size, max_len, dropout=0.1):
        super(InstEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model//2)
        self.norm_pos = nn.Embedding(100, d_model//2)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.inst_proj = nn.Linear(d_model, inst_size)
        
    def generate_src_mask(self, src):
        # src shape: (batch_size, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        x = self.embedding(src)  # Embedding
        norm_pos = self.embedding()
        # x shape: (batch_size, src_len, d_model)
        
        src_mask = self.generate_src_mask(src)  # Generate source mask
        
        # x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.inst_proj(x)
        return x  # Shape: (batch_size, src_len, inst_size)

class InstTransformer(nn.Module):
    def __init__(self, device='cpu', d_model=768, num_heads=8, d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super(InstTransformer, self).__init__()
        self.device = device
        self.chord_encoder = GPT2Model(vocab_size=150)
        self.chord_encoder.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_encoder.parameters():
            param.requires_grad = False
            
        self.inst_embedding = nn.Embedding(133, 768)
        self.init_inst_bos_proj = nn.Linear(133, 768)
        
        self.decoder = InstDecoder(d_model, num_heads, d_ff, num_layers, max_len, dropout)
        
        self.out = nn.Linear(d_model, 1)

    def forward(self, chord_tensor, init_inst_tensor, inst_idx):
        memory = self.chord_encoder(chord_tensor, return_hidden_states=True)
        
        length = memory.shape[1] - 2
        
        init_inst_embed = self.init_inst_bos_proj(init_inst_tensor)
        init_inst_embed = init_inst_embed.unsqueeze(1).repeat(1, 1, 1)
        
        # out_embed = self.inst_poolers[inst_idx](input_embed)
        inst_idx = torch.tensor([inst_idx]).to(self.device)
        # print(inst_idx.shape)
        inst_embed = self.inst_embedding(inst_idx)
        inst_embed = inst_embed.unsqueeze(1).repeat(1, length-1, 1)
        # print(inst_embed.shape)
        decoder_input_embed = torch.cat((init_inst_embed, inst_embed), dim=1)
        # print(decoder_input_embed.shape)
        decoder_output = self.decoder(decoder_input_embed, memory)
        
        out_embed = self.out(decoder_output)
        
        return out_embed
    
    def infer(self, chord_tensor, init_inst_tensor):
        # src shape: (batch_size, src_len)
        # tgt shape: (batch_size, tgt_len)
        
        # src_mask = self.generate_src_mask(src)  # Generate source mask
        # tgt_mask = self.generate_tgt_mask(tgt)  # Generate target mask

        memory = self.chord_encoder(chord_tensor, return_hidden_states=True)  # Encoder output
        # memory shape: (batch_size, src_len, d_model)
        
        length = memory.shape[1] - 2
        
        init_inst_embed = self.init_inst_bos_proj(init_inst_tensor)
        init_inst_embed = init_inst_embed.unsqueeze(1).repeat(1, 1, 1)
        
        output_container = []
        
        for inst_idx in range(133):
            # out_embed = self.inst_poolers[inst_idx](input_embed)
            inst_idx = torch.tensor([inst_idx]).to(self.device)
            # print(inst_idx.shape)
            inst_embed = self.inst_embedding(inst_idx)
            inst_embed = inst_embed.unsqueeze(1).repeat(1, length-1, 1)
            # print(inst_embed.shape)
            decoder_input_embed = torch.cat((init_inst_embed, inst_embed), dim=1)
            # print(decoder_input_embed.shape)
            decoder_output = self.decoder(decoder_input_embed, memory)
            
            out_embed = self.out(decoder_output)
            
            output_container.append(out_embed)
        
        return output_container
    
class GPT2Model(nn.Module):
    def __init__(self, vocab_size=140, n_embd=768, n_layer=12, n_head=12):
        super(GPT2Model, self).__init__()
        self.configuration = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, bos_token_id=2, eos_token_id=1)
        self.model = GPT2LMHeadModel(self.configuration)
        
    def get_embed(self, idx):
        embedding_layer = self.model.transformer.wte
        token_embedding = embedding_layer(torch.tensor([idx]))
        return token_embedding
    
    def extract_vocab_embeddings(self):
        # Extract all the embeddings for the entire vocabulary
        embedding_layer = self.model.transformer.wte
        vocab_embeddings = embedding_layer.weight.detach().clone()
        return vocab_embeddings

    def forward(self, input_ids, labels=None, return_hidden_states=False):
        attention_mask = self.make_mask(input_ids)
        # Forward pass through the transformer to get hidden states
        transformer_outputs = self.model.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Extract hidden states before the projection
        hidden_states = transformer_outputs.last_hidden_state
        
        if return_hidden_states:
            return hidden_states

        # Project the hidden states to vocabulary size
        logits = self.model.lm_head(hidden_states)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.configuration.vocab_size), labels.view(-1))
            return loss, logits
        return logits

    def make_mask(self, input_ids):
        attention_mask = (input_ids != 0).long()
        return attention_mask
    
    def infer(self, input_ids, length=2048):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        if len(input_ids.shape) > 2:
            raise Exception
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
        
        with torch.no_grad():
            for step in range(length):
                logits = self.forward(input_ids)
                output = torch.argmax(logits, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        # self.norm1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        # self.norm2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        # out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        # out = self.norm2(out)
        out += residual  # Add the residual connection
        out = self.relu(out)
        return out

class C2ITransformer(nn.Module):
    def __init__(self, mode='proj'):
        super(C2ITransformer, self).__init__()
        
        self.chord_encoder = GPT2Model(vocab_size=150)
        self.chord_encoder.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_encoder.parameters():
            param.requires_grad = False
            
        self.mode = mode
        
        self.hidden_map = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            ResidualBlock(128,128),
            nn.Linear(128,133)
        )
            
    
    def forward(self, chord_tensor):
        # Encoder
        chord_embedding = self.chord_encoder(chord_tensor, return_hidden_states=True)
        
        if self.mode == 'avg':
            pass
        
        elif self.mode == 'bos':
            chord_tensor = chord_embedding[:,0,:]
            
            output = self.hidden_map(chord_tensor)
            return output
        
    def jaccard_loss(self, pred, target):
        eps = 1e-10

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        sim = (intersection + eps) / (union + eps)
        return 1 - sim
    
    def hamming_loss(self, y_pred, y_true):
        
        # 이진화 없이 차이 계산 (torch.abs 사용)
        hamming_distance = torch.sum(torch.abs(y_true - y_pred))
        
        # 벡터 길이로 나누어 해밍 손실 계산
        hamming_loss = hamming_distance / y_true.numel()
        return hamming_loss
    
    def bce_loss(self, logits, targets):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, targets)
        return loss