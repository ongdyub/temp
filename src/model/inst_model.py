import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import Transformer
import torch.nn.functional as F
import math
import random
import numpy as np
from transformers import ASTConfig, ASTModel, GPT2Config, GPT2Model, AutoModelForCausalLM, GPT2LMHeadModel, BartConfig

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
    
class ResidualLinearLayer(nn.Module):
    def __init__(self, d_dim):
        super(ResidualLinearLayer, self).__init__()
        # Define a linear layer that keeps the dimensionality constant
        self.linear = nn.Linear(d_dim, d_dim)
        # self.down = nn.Linear(d_dim, d_dim//4)
        # self.up = nn.Linear(d_dim//4, d_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Apply the linear layer followed by an activation function
        out = self.linear(x)
        out = self.activation(out)
        # Add the input to the output for the residual connection
        return out + x

class InstPoolingLayer(nn.Module):
    def __init__(self, d_dim=1024, num_layers=8):
        super(InstPoolingLayer, self).__init__()

        self.layers = nn.ModuleList([ResidualLinearLayer(d_dim) for _ in range(num_layers)])

    def forward(self, x):
        # Sequentially apply each residual layer
        for layer in self.layers:
            x = layer(x)
        return x

class InstGRU(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=3):
        super(InstGRU, self).__init__()
        
        self.chord_transformer = GPT2Model(vocab_size=150)
        self.chord_transformer.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_transformer.parameters():
            param.requires_grad = False
            
        self.grus = nn.ModuleList([nn.GRU(1024, hidden_dim, num_layers=n_layers, batch_first=True) for _ in range(133)])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(133)])
        self.init_inst_proj = nn.Linear(133, 256)
        
    def forward(self, chord_tensor, init_inst_tensor, inst_all_container=None):
        # print("BBBBB")
        
        chord_embed = self.chord_transformer(chord_tensor, return_hidden_states=True)
        chord_embed = chord_embed[:,1:-1,:]
        length = chord_embed.shape[1]
        init_inst_embed = self.init_inst_proj(init_inst_tensor)
        init_inst_embed = init_inst_embed.unsqueeze(1).repeat(1, length, 1)
        
        input_embed = torch.cat((chord_embed, init_inst_embed), dim=2)
        
        output_container = []
        # print(input_embed.shape)
        
        for inst_idx in range(133):
            
            out_embed, hn = self.grus[inst_idx](input_embed)
            
            out_embed = self.output_layers[inst_idx](out_embed)
            
            output_container.append(out_embed)
        # print("CCCCC")
        # print(out_embed)
        # print(out_embed.shape)
        return output_container
    
    # def infer(self, chord_tensor, init_inst_tensor, length=1024):
    #     with torch.no_grad():
    #         for _ in range(length):
    #             output = self.forward(chord_tensor, init_inst_tensor)
                
    #     return

class InstPooling(nn.Module):
    def __init__(self, n_layers=2):
        super(InstPooling, self).__init__()
        
        self.chord_transformer = GPT2Model(vocab_size=150)
        self.chord_transformer.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_transformer.parameters():
            param.requires_grad = False
        
        self.init_inst_proj = nn.Linear(133, 256)
            
        self.inst_poolers = nn.ModuleList([InstPoolingLayer(num_layers=n_layers) for _ in range(133)])
        self.output_layers = nn.ModuleList([nn.Linear(1024, 1) for _ in range(133)])
        
        # self.down = nn.Linear(1024, 1024//8)
        # self.up = nn.Linear(1024//8, 1024)
        
    def forward(self, chord_tensor, init_inst_tensor, inst_all_container=None):
        # print("BBBBB")
        
        chord_embed = self.chord_transformer(chord_tensor, return_hidden_states=True)
        chord_embed = chord_embed[:,1:-1,:]
        length = chord_embed.shape[1]
        init_inst_embed = self.init_inst_proj(init_inst_tensor)
        init_inst_embed = init_inst_embed.unsqueeze(1).repeat(1, length, 1)
        
        input_embed = torch.cat((chord_embed, init_inst_embed), dim=2)
        
        output_container = []
        
        for inst_idx in range(133):
            
            out_embed = self.inst_poolers[inst_idx](input_embed)
            
            out_embed = self.output_layers[inst_idx](out_embed)
            
            output_container.append(out_embed)

        return output_container

class InstSimplePooling(nn.Module):
    def __init__(self, n_layers=2):
        super(InstSimplePooling, self).__init__()
        
        self.chord_transformer = GPT2Model(vocab_size=150)
        self.chord_transformer.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_transformer.parameters():
            param.requires_grad = False
        
        self.init_inst_proj = nn.Linear(133, 256)
            
        # self.inst_poolers = nn.ModuleList([InstPoolingLayer(num_layers=n_layers) for _ in range(133)])
        self.output_layers = nn.ModuleList([nn.Linear(1024, 1) for _ in range(133)])
        
        # self.down = nn.Linear(1024, 1024//8)
        # self.up = nn.Linear(1024//8, 1024)
        
    def forward(self, chord_tensor, init_inst_tensor, inst_all_container=None):
        # print("BBBBB")
        
        chord_embed = self.chord_transformer(chord_tensor, return_hidden_states=True)
        chord_embed = chord_embed[:,1:-1,:]
        length = chord_embed.shape[1]
        init_inst_embed = self.init_inst_proj(init_inst_tensor)
        init_inst_embed = init_inst_embed.unsqueeze(1).repeat(1, length, 1)
        
        input_embed = torch.cat((chord_embed, init_inst_embed), dim=2)
        
        output_container = []
        
        for inst_idx in range(133):
            
            # out_embed = self.inst_poolers[inst_idx](input_embed)
            
            out_embed = self.output_layers[inst_idx](input_embed)
            
            output_container.append(out_embed)

        return output_container