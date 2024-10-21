import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.expand = nn.Linear(input_dim, hidden_dim*2)
        self.dropex = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim//2)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim//4, hidden_dim//2)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(hidden_dim//2, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        # x: [bsz, 133] input binary vector
        x = x.float()
        h = F.relu(self.expand(x))
        h = self.dropex(h)
        
        h = F.relu(self.fc1(h))  # [bsz, hidden_dim]
        h = self.drop1(h)
        h = F.relu(self.fc2(h))
        h = self.drop2(h)
        h = F.relu(self.fc3(h))
        h = self.drop3(h)
        h = F.relu(self.fc4(h))

        mu = self.fc_mu(h)       # [bsz, latent_dim]
        logvar = self.fc_logvar(h)  # [bsz, latent_dim]
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.drop1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.drop2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_dim//4, hidden_dim//2)
        self.drop4 = nn.Dropout(0.2)

        self.reconstruct = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, z):
        # z: [bsz, latent_dim]
        h = F.relu(self.fc1(z))
        h = self.drop1(h)
        h = F.relu(self.fc2(h))
        h = self.drop2(h)
        h = F.relu(self.fc3(h))
        h = self.drop3(h)
        h = F.relu(self.fc4(h))
        h = self.drop4(h)
        
        x_recon = torch.sigmoid(self.reconstruct(h))  # [bsz, output_dim]
        return x_recon


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, output_dim, num_layers=3, batch_first=True)
    
    def forward(self, z, seq_lens, chord_embedding):
        # z: [bsz, latent_dim] sampled latent vector
        h = F.relu(self.fc(z)).unsqueeze(1)  # [bsz, 1, hidden_dim]
        
        # Prepare initial hidden state for GRU
        max_len = max(seq_lens)  # Maximum sequence length in the batch
        h_repeat = h.expand(-1, max_len, -1)  # [bsz, max_len, hidden_dim]
        chord_embedding = chord_embedding/100
        h_repeat = h_repeat + chord_embedding
        # Initialize an empty tensor to hold output sequences with different lengths
        outputs = []
        for i, seq_len in enumerate(seq_lens):
            # Use GRU to generate sequences of length `seq_len`
            out, _ = self.rnn(h_repeat[i:i+1, :seq_len])  # [1, seq_len, 133]
            outputs.append(out.squeeze(0))  # [seq_len, 133]
        
        return outputs  # List of tensors with shapes [seq_len_i, 133]

class VAE(nn.Module):
    def __init__(self, input_dim, encoder_hdim, decoder_hdim, latent_dim, output_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, encoder_hdim, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hdim, output_dim)
        
        self.device = device
        self.chord_encoder = GPT2Model(vocab_size=150)
        self.chord_encoder.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_encoder.parameters():
            param.requires_grad = False
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, chord_tensor, seq_lens):
        # x: [bsz, 133], seq_lens: list of output sequence lengths
        mu, logvar = self.encoder(x)  # Encode to get mu and logvar
        
        z = self.reparameterize(mu, logvar)  # Sample from the latent space
        
        chord_embedding = self.chord_encoder(chord_tensor, return_hidden_states=True)
        chord_embedding = chord_embedding[:,1:-1,:]
        output = self.decoder(z, seq_lens, chord_embedding)   # Decode to variable-length sequences
        return output, mu, logvar

# 그냥 버전에서는 인코더에 시작악기 벡터 넣을때 dim으로 보내지 말고 그냥 히든 원핫으로 넣어보는것도 생각

# 인코더 그냥 간단하게 리니어하게 z보낸 다음에 디코더로 복원할때 레이ㅓㄴ트 벡터 늘려서 rnn말고 "길이" 로 늘려서 대신에 리니어로 보낸 같은 레이턴트 벡터 길이만큼 있을때 각 벡터들 에 포지션 인코딩 느낌으로 넣는거 ㄱㅊ을듯ㄴ

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=4.0, reduction='mean'):
        """
        :param alpha: weight for class imbalance (0 < alpha < 1)
        :param gamma: focusing parameter (usually gamma >= 0)
        :param reduction: specifies the reduction to apply to the output ('none', 'mean', or 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability for the correct class

        # Apply focal loss formula
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class C2IVAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, latent_dim=64, output_dim=133):
        super(C2IVAE, self).__init__()
        
        self.chord_encoder = GPT2Model(vocab_size=150)
        self.chord_encoder.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_encoder.parameters():
            param.requires_grad = False
        
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        # Encoder layers
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of latent space
        # self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent space

        # Decoder layers
        # self.fc3 = nn.Linear(latent_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)  # Directly outputs the final shape [bsz, 133]

    def forward(self, chord_tensor):
        # Encoder
        chord_embedding = self.chord_encoder(chord_tensor, return_hidden_states=True)
        chord_tensor = chord_embedding[:,0,:]
        
        # mu, logvar = self.encode(x)
        # # Reparameterization trick
        # z = self.reparameterize(mu, logvar)
        # # Decoder
        # return self.decode(z), mu, logvar
        mu, logvar = self.encoder(chord_tensor)  # Encode to get mu and logvar
        
        z = self.reparameterize(mu, logvar)  # Sample from the latent space

        output = self.decoder(z)   # Decode to variable-length sequences
        return output, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (can be Mean Squared Error or Binary Cross Entropy)
        recon_loss = nn.BCEWithLogitsLoss(reduction='sum')(recon_x, x)
        # recon_loss = self.focal_loss(recon_x, x)
        
        # Kullback-Leibler divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss
    
    def hinge_loss(self, output, target, mu, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        margin = 1.0
        pos = (1 - target) * output  # Penalizes positive predictions for negative labels
        neg = target * (margin - output)  # Penalizes if the margin is violated for positive labels
        loss = torch.clamp(pos + neg, min=0).mean()
        return loss + kld_loss
    
    def soft_jaccard_loss(self, output, target, mu, logvar, smooth=1e-12):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        probs = torch.sigmoid(output)
        intersection = (probs * target).sum(dim=1)
        union = (probs + target - probs * target).sum(dim=1)
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou.mean() + kld_loss




import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexVAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], latent_dim=64, output_dim=133):
        super(ComplexVAE, self).__init__()
        
        self.chord_encoder = GPT2Model(vocab_size=150)
        self.chord_encoder.load_state_dict(torch.load('/workspace/out/chord_bpe/GPT2_BPE_V150/model_207_0.4520_0.3645.pt'))
        # Freeze the chord_transformer parameters
        for param in self.chord_encoder.parameters():
            param.requires_grad = False
            
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.25)
                )
            )
            in_dim = h_dim
            
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers
        hidden_dims.reverse()  # Reverse to expand back to original input size
        self.decoder_layers = nn.ModuleList()
        in_dim = latent_dim
        for h_dim in hidden_dims:
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.25)
                )
            )
            in_dim = h_dim

        # Final output layer
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
        return self.final_layer(z)

    def forward(self, chord_tensor):
        chord_embedding = self.chord_encoder(chord_tensor, return_hidden_states=True)
        x = chord_embedding[:,0,:]
        
        mu, logvar = self.encode(x)
        
        # Encoder
        # mu, logvar = self.encode(x)
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decoder
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss
