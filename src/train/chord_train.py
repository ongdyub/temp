import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..model.chord_model import ChordLSTM, ChordBiLSTM
from ..loader.chord_loader import ChordDataset, create_dataloaders

verbose = False
detail = False

# Hyper param & Dataset Setting
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 56
vocab_size = 140
num_epochs = 500
train_loader, val_loader, test_loader = create_dataloaders(batch_size)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')

# Define Model & Utils

# tmux 0
# model = ChordLSTM(vocab_size=140, embedding_dim=512, hidden_dim=512, num_layers=5).to(device)

# tmux 1
model = ChordBiLSTM(vocab_size=140, embedding_dim=512, hidden_dim=256, num_layers=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)
criterion = nn.CrossEntropyLoss()

def run(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        for input_ids in tqdm(train_loader):
            optimizer.zero_grad()
            
            inputs = input_ids[:,:-1].to(device).long()
            targets = input_ids[:, 1:].to(device).long()
            
            outputs = model(inputs)
            output_ids = torch.argmax(outputs, dim=2)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            total, correct, acc = cal_acc(output_ids, targets)
            total_cnt += total
            total_correct += correct
            total_acc += acc
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            if verbose:
                if detail:
                    print(outputs.shape)
                    print(targets.shape)
                    print(targets.view(-1).shape)
                break
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        print(f'Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f} | {total_acc / len(train_loader.dataset)}')
        
        
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for input_ids in tqdm(val_loader):
                inputs = input_ids[:,:-1].to(device).long()
                targets = input_ids[:, 1:].to(device).long()
                
                outputs = model(inputs)
                output_ids = torch.argmax(outputs, dim=2)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                total, correct, acc = cal_acc(output_ids, targets)
                total_cnt += total
                total_correct += correct
                total_acc += acc
                
                total_val_loss += loss.item()
                
                if verbose:
                    if detail:
                        print(outputs.shape)
                        print(targets.shape)
                        print(targets.view(-1).shape)
                    break

        val_loss.append(total_val_loss / len(val_loader.dataset))
        
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f} | {total_acc / len(val_loader.dataset)}')

        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            patience_counter = 0

            torch.save(model.state_dict(), f'../../../workspace/out/chordbi/model_{epoch+1}_{val_loss[-1]:.6f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]}')
            
            with open(f'../../../workspace/out/chordbi/train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open(f'../../../workspace/out/chordbi/val_loss.pkl', 'wb') as file:
                pickle.dump(val_loss, file)
            
        else:
            patience_counter += 1
            
            with open(f'../../../workspace/out/chordbi/train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open(f'../../../workspace/out/chordbi/val_loss.pkl', 'wb') as file:
                pickle.dump(val_loss, file)
        
        
        
def inference_next(input_ids):
    pass
    return next_ids        

def cal_acc(infer, target):
    if infer.shape[0] != target.shape[0] or infer.shape[1] != target.shape[1]:
        raise Exception
    
    if verbose:
        if detail:
            print("CAL ACC")
            print(infer.shape)
            print(target.shape)
        
    correct = 0
    total = 0
    
    for i in range(infer.shape[0]):
            
        zero_infer = torch.where(infer[i] == 0)[0]
        zero_target = torch.where(target[i] == 0)[0]
        
        zero_infer = zero_infer[0].item() if len(zero_infer) > 0 else 2047
        zero_target = zero_target[0].item() if len(zero_target) > 0 else 2047
        
        max_zero = max(zero_infer, zero_target)
        
        if verbose:
            if detail:
                print("IN FOR")
                print(infer[i].shape)
                print(infer[i][:10])
                print(infer[i][-10:])
                print(target[i].shape)
                print(target[i][:10])
                print(target[i][-10:])
                print(zero_infer)
                print(zero_target)
                print(max_zero)
        
        infer_slice = infer[i][:max_zero]
        target_slice = target[i][:max_zero]
        
        correct += (infer_slice == target_slice).sum().item()
        total += max_zero

    return total, correct, correct/total
        
        
        
        
        
run()