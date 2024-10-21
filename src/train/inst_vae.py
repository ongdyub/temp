import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..model.inst_vae import VAE
from ..loader.inst_loader import InstVAEDataset, create_VAE
import torch.nn.functional as F
import setproctitle

verbose = False
detail = False

device = torch.device("cuda:11" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 32
num_epochs = 500
#############################################################
#############################################################
#############################################################


#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
folder = 'inst_vae'
for i in range(1):
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    base_model = 'VAE'
    # Usage example:
    input_dim = 133    # Input dimension
    encoder_hdim = 512   # Hidden layer size
    decoder_hdim = 768
    latent_dim = 128    # Latent space dimension
    output_dim = 133   # Output dimension (sequence element size)

    model = VAE(input_dim, encoder_hdim, decoder_hdim, latent_dim, output_dim, device)
    name = f'{base_model}_Vanila'
    
    name = f'{base_model}_POS_10'
    
    name = f'{base_model}_Focal_1_2'
    
    name = f'{base_model}_Focal_0.25_3'
    
    name = f'{base_model}_Focal_0.25_2'


for i in range(1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


train_loader, val_loader, test_loader = create_VAE(batch_size)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')
print(name)
setproctitle.setproctitle(name)

if not os.path.exists(f'../../../workspace/out/{folder}/{name}'):
    os.makedirs(f'../../../workspace/out/{folder}/{name}')


def run(type, model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    best_test_acc = -1
    best_val_acc = -1
    patience_counter = 0
    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []
    test_acc = []
    
    if type == 'VAE':
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            total_recon_loss = 0
            total_kld_loss = 0
        
            total_cnt = 0
            total_correct = 0
            total_candidate = 0
            
            for chord, init, inst, length in tqdm(train_loader, ncols=60):
                
                optimizer.zero_grad()
                
                chord = chord.to(device)
                init = init.to(device)
                target = inst.to(device)
                
                recon_x, mu, logvar = model(init, chord, length)
                # print(length)
                # print(target)
                # print(target.shape)
                loss, recon_loss, kld_loss, correct, cnt = vae_loss_function(recon_x, target, mu, logvar, length)
                loss.backward()
                        
                total_train_loss += loss.item()
                total_recon_loss += recon_loss
                total_kld_loss += kld_loss
                
                if cnt != (sum(length)*133):
                    print(cnt)
                    print(sum(length)*133)
                    print("ACC Calculate ERROR")
                    break
                
                total_correct += correct
                total_candidate += cnt
                
                total_cnt += 1
                    
                optimizer.step()
                # break
            
            total_train_acc = total_correct/total_candidate
            scheduler.step()
            train_loss.append(total_train_loss / total_cnt)
            train_acc.append(total_train_acc)
            print(f'TRN Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss / total_cnt:.8f} Acc: {int(total_correct)} / {total_candidate} : {total_train_acc:.8f}')
            # print(f'EPO {epoch} RECON {total_recon_loss/total_cnt} KLD {total_kld_loss/total_cnt} TOTAL {total_train_loss/total_cnt}')
            
            
            model.eval()
            
            total_train_loss = 0
            total_recon_loss = 0
            total_kld_loss = 0
        
            total_cnt = 0
            total_correct = 0
            total_candidate = 0
            
            with torch.no_grad():
                for chord, init, inst, length in tqdm(val_loader, ncols=60):
                    chord = chord.to(device)
                    init = init.to(device)
                    target = inst.to(device)
                    
                    recon_x, mu, logvar = model(init, chord, length)
                    loss, recon_loss, kld_loss, correct, cnt = vae_loss_function(recon_x, target, mu, logvar, length)
                            
                    total_train_loss += loss.item()
                    total_recon_loss += recon_loss
                    total_kld_loss += kld_loss
                    
                    if cnt != (sum(length)*133):
                        print(cnt)
                        print(sum(length)*133)
                        print("ACC Calculate ERROR")
                        break
                    
                    total_correct += correct
                    total_candidate += cnt
                    
                    total_cnt += 1
                    # break
            
            total_train_acc = total_correct/total_candidate
            val_loss.append(total_train_loss / total_cnt)
            val_acc.append(total_train_acc)
            print(f'VAL Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss / total_cnt:.8f} Acc: {int(total_correct)} / {total_candidate} : {total_train_acc:.8f}')
            
            total_train_loss = 0
            total_recon_loss = 0
            total_kld_loss = 0
        
            total_cnt = 0
            total_correct = 0
            total_candidate = 0
            
            with torch.no_grad():
                for chord, init, inst, length in tqdm(test_loader, ncols=60):
                    chord = chord.to(device)
                    init = init.to(device)
                    target = inst.to(device)
                    
                    recon_x, mu, logvar = model(init, chord, length)
                    loss, recon_loss, kld_loss, correct, cnt = vae_loss_function(recon_x, target, mu, logvar, length)
                            
                    total_train_loss += loss.item()
                    total_recon_loss += recon_loss
                    total_kld_loss += kld_loss
                    
                    if cnt != (sum(length)*133):
                        print(cnt)
                        print(sum(length)*133)
                        print("ACC Calculate ERROR")
                        break
                    
                    total_correct += correct
                    total_candidate += cnt
                    
                    total_cnt += 1
                    # break
            
            total_train_acc = total_correct/total_candidate
            test_acc.append(total_train_acc)
            print(f'TST Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss / total_cnt:.8f} Acc: {int(total_correct)} / {total_candidate} : {total_train_acc:.8f}')
            
            
            with open(f'../../../workspace/out/{folder}/{name}/train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open(f'../../../workspace/out/{folder}/{name}/val_loss.pkl', 'wb') as file:
                pickle.dump(val_loss, file)
            with open(f'../../../workspace/out/{folder}/{name}/train_acc.pkl', 'wb') as file:
                pickle.dump(train_acc, file)
            with open(f'../../../workspace/out/{folder}/{name}/val_acc.pkl', 'wb') as file:
                pickle.dump(val_acc, file)
            with open(f'../../../workspace/out/{folder}/{name}/test_acc.pkl', 'wb') as file:
                pickle.dump(test_acc, file)

            if test_acc[-1] > best_test_acc or train_acc[-1] > 0.8:
                best_test_acc = test_acc[-1]
                patience_counter = 0

                torch.save(model.state_dict(), f'../../../workspace/out/{folder}/{name}/model_{epoch+1}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')
                print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}, Test Acc: {test_acc[-1]:.6f}')        
            else:
                patience_counter += 1
                
            if patience_counter == 5:
                print("Early Stop")
                print("Finish The Train Model")
                    
                    
# Function to calculate accuracy
def calculate_accuracy(output, target):
    output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
    predictions = (output > 0.3).float()  # Convert probabilities to binary (0 or 1)
    correct = (predictions == target).float().sum()
    # accuracy = correct / target.numel()
    # return accuracy.item()
    return correct, target.numel()

# def save_epoch_loss(epoch, epoch_loss):
#     if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
#         os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
#     with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
#             pickle.dump(epoch_loss, file)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for the class imbalance. You can adjust this based on your dataset.
            gamma: Focusing parameter to reduce the relative loss for well-classified examples.
            reduction: Specifies the reduction to apply to the output ('mean', 'sum', 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Sigmoid activation to get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        pt = torch.exp(-bce_loss)  # pt is the probability of the target class
        
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Reduce the loss based on the selected reduction mode
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)
# Loss function for VAE: combines reconstruction loss and KL-divergence
def vae_loss_function(recon_x, x, mu, logvar, seq_lens):
    # recon_x: list of [seq_len_i, 133] (decoded sequences)
    # x: 
    # seq_lens: list of sequence lengths
    # mu, logvar: VAE latent distribution parameters

    # Reconstruction loss: Binary Cross-Entropy (or can use MSE)
    recon_loss = 0
    
    correct_total = 0
    cnt_total = 0
    
    for i, seq_len in enumerate(seq_lens):
        # Repeat the input across the sequence length to compare with output
        x_repeated = x[i, :seq_len, :]  # [seq_len, 133]
        # print(x_repeated.shape)
        
        ######
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(recon_x[i])
        
        # Adjust target based on custom threshold (e.g., 0.3 instead of 0.5)
        thresholded_output = (probs > 0.5).float()
        ######
        # recon_loss += nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))(recon_x[i], x_repeated)  # Loss for this sequence
        # recon_loss += nn.BCEWithLogitsLoss()(recon_x[i], x_repeated)  # Loss for this sequence
        recon_loss += focal_loss_fn(recon_x[i], x_repeated)
        
        correct, total = calculate_accuracy(recon_x[i], x_repeated)
        correct_total += correct
        cnt_total += total

    # KL-divergence loss: encourages latent variables to follow normal distribution
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss is sum of reconstruction and KL-divergence loss
    total_loss = recon_loss + kld_loss
    return total_loss, recon_loss, kld_loss, correct_total, cnt_total





run(base_model)
