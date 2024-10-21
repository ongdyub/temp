import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from ..model.inst_transformer import InstTransformer, InstEncoder, InstNoPEEncoder, InstTransformer, Transformer
from ..loader.inst_loader import InstDataset, create_dataloaders, create_Group
import setproctitle

verbose = False
detail = False

device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 4
src_vocab_size = 150
tgt_vocab_size = 68213
num_epochs = 500
#############################################################
#############################################################
#############################################################

folder = 'inst'
for i in range(1):
    name = f'GroupTrans_Unk'
    mode = 'Group'
    model = Transformer(src_vocab_size=150, tgt_vocab_size=68213)

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


# name = f'Test'
# model = InstNoPEEncoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
# Optimizer
for i in range(1):
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

if mode == 'Group':
    train_loader, val_loader, test_loader = create_Group(batch_size)
else:
    train_loader, val_loader, test_loader = create_dataloaders(batch_size)
    
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
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        epoch_loss = []
        
        for (chords, targets, lengths) in tqdm(train_loader, ncols=60):
            optimizer.zero_grad()
        
            chords = chords.to(device)
            inst_inputs = targets[:, :-1].to(device)
            inst_targets = targets[:, 1:].to(device)
            
            outputs = model(chords, inst_inputs)
            
            loss = criterion(outputs.view(-1, tgt_vocab_size), inst_targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            output_ids = torch.argmax(outputs, dim=2)
            total, correct, acc = cal_acc(output_ids, inst_targets)
            
            total_cnt += total
            total_correct += correct
            total_acc += acc
            
            total_train_loss += loss.item()
            epoch_loss.append(loss.item())

            if verbose:
                break
        
        scheduler.step()
        save_epoch_loss(epoch, epoch_loss)
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.8f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}, INFO: {name}')
        
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for (chords, targets, lengths) in tqdm(val_loader, ncols=60):
                
                chords = chords.to(device)
                inst_inputs = targets[:, :-1].to(device)
                inst_targets = targets[:, 1:].to(device)
                
                outputs = model(chords, inst_inputs)
                output_ids = torch.argmax(outputs, dim=2)
                loss = criterion(outputs.view(-1, tgt_vocab_size), inst_targets.reshape(-1))
                
                total_val_loss += loss.item()
                
                total, correct, acc = cal_acc(output_ids, inst_targets)
            
                total_cnt += total
                total_correct += correct
                total_acc += acc
                
                if verbose:
                    break

        val_loss.append(total_val_loss / len(val_loader.dataset))
        val_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.8f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.6f}') 
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        for (chords, targets, lengths) in tqdm(test_loader, ncols=60):
            seq_len = targets.shape[1]
            if seq_len < 12:
                input_len = 4
                max_len = seq_len
            elif seq_len < 100:
                input_len = 8
                max_len = seq_len
            else:
                input_len = 8
                max_len = 100
            
            chords = chords.to(device)
            
            inst_inputs = targets[:,:input_len].to(device)
            inst_targets = targets[:,:max_len].to(device)
            
            out = model.infer(chords, inst_inputs, length=max_len)
            out = out[:,:max_len]
            
            total, correct, acc = cal_acc(out, inst_targets)

            total_cnt += total
            total_correct += correct
            total_acc += acc
            
            if verbose:
                break
        
        test_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Test Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.6f}')
            
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
      

def cal_acc(infer, target):
    if infer.shape[0] != target.shape[0] or infer.shape[1] != target.shape[1]:
        raise Exception
    correct = 0
    total = 0
    
    for i in range(infer.shape[0]):
        zero_infer = torch.where(infer[i] == 1)[0]
        zero_target = torch.where(target[i] == 1)[0]
        zero_target = zero_target[0].item() if len(zero_target) > 0 else target[i].shape[0]

        max_zero = zero_target
        
        infer_slice = infer[i][:max_zero]
        target_slice = target[i][:max_zero]
        
        correct += (infer_slice == target_slice).sum().item()
        total += max_zero
    if total == 0.0 or total == 0:
        total = 1

    return total, correct, correct/total

        
def save_epoch_loss(epoch, epoch_loss):
    if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
        os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
    with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
            pickle.dump(epoch_loss, file)

run()
