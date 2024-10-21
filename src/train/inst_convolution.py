import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..model.inst_conv import InstTFEncoderConvolution
from ..loader.inst_loader import InstVAEDataset, create_VAE
import torch.nn.functional as F
import setproctitle

verbose = True
detail = False

device = torch.device("cuda:14" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 2
num_epochs = 500
#############################################################
#############################################################
#############################################################


#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
folder = 'inst_conv'
for i in range(1):
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    base_model = 'InstTFEncoderConvolution'
    # Usage example:
    input_dim = 133    # Input dimension
    encoder_hdim = 512   # Hidden layer size
    decoder_hdim = 768
    latent_dim = 128    # Latent space dimension
    output_dim = 133   # Output dimension (sequence element size)

    model = InstTFEncoderConvolution()
    name = 'T_Encoder'


for i in range(1):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
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
    padding_idx = -1
    
    best_test_acc = -1
    best_val_acc = -1
    patience_counter = 0
    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []
    test_acc = []
    
    if type != 'VAE':
        
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
                
                mask = (target != padding_idx).float()
                target = torch.where(target == padding_idx, torch.zeros_like(target), target)
                
                output = model(chord)

                # Calculate BCEWithLogits loss per element
                loss_per_element = bce_loss_fn(output, target)

                # Apply the mask to the loss (only consider valid positions)
                loss_per_element = loss_per_element * mask

                # Calculate the mean loss, considering only the valid positions (ignoring padding)
                loss = loss_per_element.sum() / mask.sum()

                loss.backward()
                
                
                correct, cnt = conv_acc(output, target, length)
                
                total_train_loss += loss.item()
                
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
                    
                    mask = (target != padding_idx).float()
                    target = torch.where(target == padding_idx, torch.zeros_like(target), target)
                    
                    output = model(chord)

                    # Calculate BCEWithLogits loss per element
                    loss_per_element = bce_loss_fn(output, target)

                    # Apply the mask to the loss (only consider valid positions)
                    loss_per_element = loss_per_element * mask

                    # Calculate the mean loss, considering only the valid positions (ignoring padding)
                    loss = loss_per_element.sum() / mask.sum()
                    
                    correct, cnt = conv_acc(output, target, length)
                    
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
                    
                    mask = (target != padding_idx).float()
                    target = torch.where(target == padding_idx, torch.zeros_like(target), target)
                    
                    output = model(chord)

                    # Calculate BCEWithLogits loss per element
                    loss_per_element = bce_loss_fn(output, target)

                    # Apply the mask to the loss (only consider valid positions)
                    loss_per_element = loss_per_element * mask

                    # Calculate the mean loss, considering only the valid positions (ignoring padding)
                    loss = loss_per_element.sum() / mask.sum()
                    
                    correct, cnt = conv_acc(output, target, length)
                    
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
    correct = (output == target).float().sum()
    # accuracy = correct / target.numel()
    # return accuracy.item()
    return correct, target.numel()

# def save_epoch_loss(epoch, epoch_loss):
#     if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
#         os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
#     with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
#             pickle.dump(epoch_loss, file)

def conv_acc(output, target, seq_lens):
    # recon_x: list of [seq_len_i, 133] (decoded sequences)
    # x: 
    # seq_lens: list of sequence lengths
    
    correct_total = 0
    cnt_total = 0
    
    for i, seq_len in enumerate(seq_lens):
        # print(seq_lens)
        # print(seq_len)
        infer_out = torch.sigmoid(output[i])
        
        # Adjust target based on custom threshold (e.g., 0.3 instead of 0.5)
        infer_out = (infer_out > 0.5).float()
        
        #adjust lengs
        infer_out = infer_out[:seq_len, :]
        # print(infer_out.shape)
        # print(target.shape)
        # print(i)
        target_ = target[i, :seq_len, :]
        # print(target.shape)
        
        correct, total = calculate_accuracy(infer_out, target_)
        correct_total += correct
        cnt_total += total

    return correct_total, cnt_total

run(base_model)
