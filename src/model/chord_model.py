import math
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import T5Config, T5ForConditionalGeneration

class ChordLSTM(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.01, eps=1e-9)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0.001)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(output)
        return logits
    
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
                output = self.forward(input_ids)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super(PositionalEncoding, self).__init__()

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return token_embedding + self.pos_encoding[:,:token_embedding.size(1)]


class ChordTransformer(nn.Module):
    # Constructor
    def __init__( self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, ):
        super(ChordTransformer, self).__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
        # UTILS
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, eps=1e-9)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-9)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
    # def forward(self, src, src_mask):
        # Src, Tgt size 는 반드시 (batch_size, src sequence length) 여야 합니다.

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        output = self.out(transformer_out)
        
        # src = self.embedding(src) * math.sqrt(self.dim_model)
        # src = self.positional_encoder(src)
        
        # output = self.transformer(src, src, src_mask)
        # output = self.out(output)

        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=next(self.parameters()).device) * float('-inf'), diagonal=1)
        return mask

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)
    
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
                # src_mask = self.generate_square_subsequent_mask(input_ids.size(1))                
                inputs = input_ids[:,:]
                targets = input_ids[:, 1:]
                
                sequence_length = targets.size(1)
                tgt_mask = self.get_tgt_mask(sequence_length)
                
                output = self.forward(input_ids, targets, tgt_mask)
            
                # output = self.forward(input_ids, src_mask)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids

    
# Initialize the model
# model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)


class ChordT5(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordT5, self).__init__()
        self.configuration = T5Config(vocab_size=vocab_size, pad_token_id=1, eos_token_id=0, decoder_start_token_id=2)
        self.transformer = T5ForConditionalGeneration(self.configuration)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, eps=1e-9)

    # def forward(self, input_ids, decode_ids):
    #     output = self.transformer(input_ids=input_ids, decoder_input_ids=decode_ids)
    #     return output
    
    def forward(self, input_ids, labels=None):
        output = self.transformer(input_ids=input_ids, labels=labels)
        return output
    
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
                # src_mask = self.generate_square_subsequent_mask(input_ids.size(1))                
                inputs = input_ids[:,:]
                targets = input_ids[:, 1:]
                
                sequence_length = targets.size(1)
                tgt_mask = self.get_tgt_mask(sequence_length)
                
                output = self.forward(input_ids, targets, tgt_mask)
            
                # output = self.forward(input_ids, src_mask)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((input_ids, predict), dim=-1)

                input_ids = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids