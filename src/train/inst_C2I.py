
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from ..model.inst_transformer import InstTransformer, InstEncoder, InstNoPEEncoder, InstTransformer, C2IEncoder, C2ITransformer
from ..model.inst_vae import C2IVAE, ComplexVAE
from ..loader.inst_loader import C2IDataset, create_C2I
import setproctitle

import torch
import torch.nn.functional as F


verbose = False
detail = False

device = torch.device("cuda:13" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 32
vocab_size = 150
num_epochs = 500
max_len=5000
dropout=0.1
#############################################################
#############################################################
#############################################################
if verbose:
    folder = 'VER_inst_TF'
else:
    folder = 'inst_TF'
    
for i in range(1):
    name = f'Proj_LEN_1'
    d_model = 512
    num_heads = 8
    d_ff = d_model*4
    num_layers = 8
    model = C2IEncoder(d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, vocab_size=vocab_size, max_len=max_len, dropout=0.1)
    
    name = f'Avg_LEN_1'
    
    model = C2ITransformer(mode='bos')
    name = 'BOS_BCE'
    
    # model = C2IVAE()
    # name = f'VAE_Focal_0.1_5'
    # name = f'VAE_Focal_0.3_5'
    # name = f'VAE_Focal_0.3_2'
    # name = f'VAE_Focal_0.05_10'
    # name = f'VAE_Focal_0.5_2'
    
    
    # name = 'HINGE_LRUP'
    # name = "JARCCARD_LRUP"
    # name = 'BCE_sum'

# POS weight coordinate
while False:
    # pos_weight = torch.tensor([       46188,        46188,        46188,        46188, 7.9315e-01, 6.2884e+01,
    #     1.5296e+02, 1.3405e+02, 6.4145e+01, 2.3586e+02, 5.2582e+01, 2.9319e+02,
    #     3.5541e+01, 3.0591e+00, 6.2271e+01, 6.5360e+00, 6.8859e+00, 8.3859e+00,
    #     6.4210e+00, 1.9471e+02, 1.6106e+02, 2.0069e+02, 1.5882e+02, 2.8513e+01,
    #     1.6574e+02, 6.2445e+01, 7.9049e+01, 3.7759e+02, 2.4061e+01, 2.5637e+01,
    #     8.1185e+01, 2.1367e+01, 1.1030e+02, 6.3780e+01, 6.9840e+01, 4.0063e+02,
    #     9.1983e+00, 1.1777e+01, 2.0062e+01, 1.2911e+02, 3.1536e+02, 3.1754e+02,
    #     1.2250e+02, 2.0247e+02, 1.7229e+00, 3.0623e+00, 2.2435e+00, 2.2968e+00,
    #     3.2494e+01, 4.2738e+00, 3.9982e+00, 7.6384e-01, 1.3991e+00, 4.4685e+01,
    #     4.3886e+01, 1.6944e+02, 5.9844e+00, 1.0164e+02, 2.4209e+02, 2.2106e+02,
    #     2.5103e-01, 5.1302e-01, 7.8229e-01, 2.7940e+01, 3.3816e-01, 2.8741e+01,
    #     8.1039e+01, 2.0428e+02, 2.0423e+01, 1.8560e+00, 2.3387e+00, 3.2189e+00,
    #     6.1091e-01, 1.1261e+01, 7.9545e-01, 3.4526e-01, 2.3892e+00, 1.9531e-01,
    #     4.9868e+01, 7.8772e+01, 4.1139e+02, 4.9036e+02, 1.0976e+02, 8.0460e+01,
    #     6.8040e+01, 6.7427e+01, 2.4082e+02, 5.4886e+02, 1.2473e+03, 2.5560e+02,
    #     2.0072e+03, 1.2802e+02, 1.6337e+02, 1.7134e+02, 3.7148e+02, 1.8302e+02,
    #     3.6850e+02, 3.8390e+02, 3.5429e+02, 5.0104e+02, 1.7802e+02, 6.9882e+02,
    #     4.8012e+02, 2.9892e+02, 3.0488e+02, 8.3878e+02, 5.9115e+02, 5.4239e+02,
    #     3.1975e+02, 1.0717e+02, 7.2069e+02, 2.3956e+02, 3.8390e+02, 2.2321e+02,
    #     3.1536e+02, 1.1833e+03, 1.3366e+02, 3.0782e+03, 2.3465e+02, 7.9534e+02,
    #     1.8084e+02, 1.8830e+02, 1.5917e+03, 1.9809e+02, 4.1979e+03, 1.5917e+03,
    #     7.0958e+02, 2.3084e+03, 3.5519e+03, 2.4299e+03, 3.2981e+03, 8.8723e+02,
    #     5.1774e-01]).to(device)
    pass


for i in range(1):
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    # criterion = nn.BCEWithLogitsLoss()
    
    
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = FocalLoss(alpha=1, gamma=2)
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


train_loader, val_loader, test_loader = create_C2I(batch_size)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')
print(name)

setproctitle.setproctitle(name)

if not os.path.exists(f'../../../workspace/out/{folder}/{name}'):
    os.makedirs(f'../../../workspace/out/{folder}/{name}')


def run(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    best_test_acc = -1
    best_val_acc = -1
    patience_counter = 0
    
    patience_counter = 0
    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        total_z_z = 0
        total_o_o = 0
        total_lack = 0
        total_extra = 0
        total_acc = 0
        total_cnt = 0
        
        epoch_loss = []
        
        for (chords, targets, lengths) in tqdm(train_loader, ncols=60):
            optimizer.zero_grad()

            chords = chords.to(device)
            targets = targets.to(device)
            
            outputs = model(chords)
            # outputs = model.proj_inst(outputs, length=lengths.to(device))
            # outputs = model.avg_inst(outputs, length=lengths.to(device))
            # outputs, mu, logvar = model(chords)

            # loss = criterion(outputs, targets)
            # loss = model.loss_function(outputs, targets, mu, logvar)
            # loss = model.hinge_loss(outputs, targets, mu, logvar)
            # loss = model.soft_jaccard_loss(outputs, targets, mu, logvar)
            loss = model.bce_loss(outputs, targets)
            
            loss.backward()
            
            optimizer.step()
            
            total_train_loss += loss.item()
            epoch_loss.append(loss.item())
            
            z_z, o_o, extra, lack = subset_accuracy(outputs, targets, lengths)
            
            total_cnt += sum(lengths)
            total_z_z += z_z
            total_o_o += o_o
            total_lack += lack
            total_extra += extra
            total_acc += (o_o + z_z) / (133*sum(lengths))
            
            if verbose:
                break

        scheduler.step()
        save_epoch_loss(epoch, epoch_loss)
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        train_acc.append((total_o_o) / (total_o_o+total_extra+total_lack))
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.8f}, 1_1_Accuracy: {total_o_o}/{total_o_o+total_extra+total_lack} {((total_o_o) / (total_o_o+total_extra+total_lack)):.8f}, 0_0: {total_z_z}, 1_1: {total_o_o} lack: {total_lack} extra: {total_extra} LR: {scheduler.get_last_lr()[0]:.8f}, INFO: {name}')
        
        model.eval()
        
        total_val_loss = 0
        
        total_z_z = 0
        total_o_o = 0
        total_lack = 0
        total_extra = 0
        total_acc = 0
        total_cnt = 0
        
        with torch.no_grad():
            for (chords, targets, lengths) in tqdm(val_loader, ncols=60):
                
                chords = chords.to(device)
                targets = targets.to(device)
                    
                outputs = model(chords)
                # outputs = model.proj_inst(outputs, length=lengths.to(device))
                # outputs = model.avg_inst(outputs, length=lengths.to(device))
                # outputs, mu, logvar = model(chords)
                

                # loss = criterion(outputs, targets)
                # loss = model.loss_function(outputs, targets, mu, logvar)
                # loss = model.hinge_loss(outputs, targets, mu, logvar)
                # loss = criterion(outputs, targets)
                # loss = model.soft_jaccard_loss(outputs, targets, mu, logvar)
                
                loss = model.bce_loss(outputs, targets)
                
                total_val_loss += loss.item()
                
                z_z, o_o, extra, lack = subset_accuracy(outputs, targets, lengths)
            
                total_cnt += sum(lengths)
                total_z_z += z_z
                total_o_o += o_o
                total_lack += lack
                total_extra += extra
                total_acc += (o_o + z_z) / (133*sum(lengths))
                
                if verbose:
                    break

        val_loss.append(total_val_loss / len(val_loader.dataset))
        val_acc.append((total_o_o) / (total_o_o+total_extra+total_lack))
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, 1_1_Accuracy: {total_o_o}/{total_o_o+total_extra+total_lack} {((total_o_o) / (total_o_o+total_extra+total_lack)):.8f}, 0_0: {total_z_z}, 1_1: {total_o_o} lack: {total_lack} extra: {total_extra}') 
        
        total_z_z = 0
        total_o_o = 0
        total_lack = 0
        total_extra = 0
        total_acc = 0
        total_cnt = 0
        
        for (chords, targets, lengths) in tqdm(test_loader, ncols=60):
            chords = chords.to(device)
            targets = targets.to(device)
            
            outputs = model(chords)
            # outputs = model.proj_inst(outputs, length=lengths.to(device))
            # outputs = model.avg_inst(outputs, length=lengths.to(device))
            # outputs, mu, logvar = model(chords)

            # loss = criterion(outputs, targets)
            # loss = model.loss_function(outputs, targets, mu, logvar)
            # loss = criterion(outputs, targets)
            # loss = model.hinge_loss(outputs, targets, mu, logvar)
            # loss = model.soft_jaccard_loss(outputs, targets, mu, logvar)
            loss = model.bce_loss(outputs, targets)
                
            z_z, o_o, extra, lack = subset_accuracy(outputs, targets, lengths)
            
            total_cnt += sum(lengths)
            total_z_z += z_z
            total_o_o += o_o
            total_lack += lack
            total_extra += extra
            total_acc += (o_o + z_z) / (133*sum(lengths))
            
            if verbose:
                break
        
        test_acc.append((total_o_o) / (total_o_o+total_extra+total_lack))
        print(f'Epoch {epoch+1}, Test 1_1_Accuracy: {total_o_o}/{total_o_o+total_extra+total_lack} {((total_o_o) / (total_o_o+total_extra+total_lack)):.8f}, 0_0: {total_z_z}, 1_1: {total_o_o} lack: {total_lack} extra: {total_extra}')
            
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

        if test_acc[-1] > best_test_acc or val_acc[-1] > best_val_acc:
            if test_acc[-1] > best_test_acc:
                best_test_acc = test_acc[-1]
            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
            patience_counter = 0
            
            torch.save(model.state_dict(), f'../../../workspace/out/{folder}/{name}/model_{epoch+1}_{train_acc[-1]:.4f}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')

            print(f'Model saved: Epoch {epoch+1} with Train Loss: {train_loss[-1]:.6f}, Val Loss: {val_loss[-1]:.6f}')        
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}, Test Acc: {test_acc[-1]:.6f}')
        else:
            patience_counter += 1
            
        if patience_counter == 5:
            print("Early Stop")
            print("Finish The Train Model")
      
# 일단 하나씩 돌거다
# 아웃풋 뽑고
# 유효한 길이로 먼저 자르고
# 전체 마디가 있으렌데
# 각 마디별로
# 현재 마디에 나와야 하는 악기의 수
# 정확히 맞춘 수
# 나와야하는데 못맞춘거 / 안나와야하는데 맞춘거

def subset_accuracy(outputs, targets, lengths):
    # Apply sigmoid to outputs to get probabilities
    probs = torch.sigmoid(outputs)  # probs: [batch_size, max_length, num_classes]
    
    # Convert probabilities to binary predictions
    preds = (probs > 0.5).float()  # preds: [batch_size, max_length, num_classes]
    
    # ones_tensor = torch.ones_like(targets)
    # zeros_tensor = torch.zeros_like(targets)
    # Calculate correct, extra, and lack predictions with masking
    # correct = (preds == targets)
    # z_z = ((correct == torch.logical_not(targets)) & mask).sum().item()
    # o_o = ((correct == targets) & mask).sum().item()
    # extra = ((preds == targets + 1) & mask).sum().item()
    # lack = ((preds == targets - 1) & mask).sum().item()
    
    # Target - Pred    
    zero_zero = torch.sum(((targets == 0) & (preds == 0))).item()
    zero_one = torch.sum(((targets == 0) & (preds == 1))).item()
    one_zero = torch.sum(((targets == 1) & (preds == 0))).item()
    one_one = torch.sum(((targets == 1) & (preds == 1))).item()
    
    return zero_zero, one_one, zero_one, one_zero

        
def save_epoch_loss(epoch, epoch_loss):
    if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
        os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
    with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
            pickle.dump(epoch_loss, file)


run()
