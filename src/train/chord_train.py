import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from ..model.chord_model import ChordLSTM, ChordTransformer, ChordT5
from ..loader.chord_loader import ChordDataset, create_dataloaders

verbose = True
detail = False

# Hyper param & Dataset Setting
#############################################################
#############################################################
#############################################################
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 2
batch_size = 32
# batch_size = 96
batch_size = 2

vocab_size = 150
vocab_size = 406070
vocab_size = 15000
num_epochs = 200
#############################################################
#############################################################
#############################################################



train_loader, val_loader, test_loader = create_dataloaders(batch_size)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')

#############################################################
#############################################################
#############################################################
# Define Model & Utils
# tmux 0
# model = ChordLSTM(vocab_size=140, embedding_dim=512, hidden_dim=512, num_layers=5).to(device)
# name = 'LSTM'
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)
# criterion = nn.CrossEntropyLoss()

# tmux X
# model = ChordTransformer(num_tokens=140, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1).to(device)
# name = 'TRANS'

# tmux 2
# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'LSTM_G1'


# tmux 1
# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=128, num_layers=3)
# name = 'LSTM_G3'

# tmux 0
# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'LSTM_G2'

# tmux 3
# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'LSTM_G2_STRIDE'

# tmux 4?
model = ChordT5(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
name = 'T5'

optimizer = model.optimizer
# scheduler = model.scheduler
criterion = nn.CrossEntropyLoss()

# Check if CUDA is available and wrap the model with DataParallel
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
#     model.cuda()  # Move the model to GPU before wrapping with DataParallel
#     device_ids = [0, 1, 2, 3]
#     model = nn.DataParallel(model, device_ids=device_ids)
# else:
#     print("CUDA is not available. Training on CPU.")


#############################################################
#############################################################
#############################################################

def run(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    best_val_acc = -1
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
        
        for (inputs, targets) in tqdm(train_loader, ncols=60):
            if inputs.size(0) == 0:
                print("Skip Zero-Size")
                continue
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Transformer
            # 다음 단어를 마스킹하려면 마스크 가져오기
            # sequence_length = targets.size(1)
            # tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            # outputs = model(inputs, targets, tgt_mask)
            
            # LSTM
            # outputs = model(inputs)
            
            # T5
            outputs = model(inputs, labels=targets)
            
            # output_ids = torch.argmax(outputs, dim=2)
            
            #T5
            output_ids = torch.argmax(outputs.logits, dim=2)
            
            # loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss = criterion(outputs.logits.view(-1, vocab_size), targets.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Stopping training, encountered {loss.item()} loss at epoch {epoch}")
                loss = torch.clamp(loss, min=0.0001)
                continue
            
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            total, correct, acc = cal_acc(output_ids, targets)
            total_cnt += total
            total_correct += correct
            total_acc += acc

            total_train_loss += loss.item()
            
            if verbose:
                if detail:
                    print("Train")
                    print(inputs[:,:5])
                    print(targets[:,:5])
                    print(outputs.shape)
                    print(targets.shape)
                    print(targets.view(-1).shape)
                break
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        # train_loss.append(total_train_loss)
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: scheduler.get_last_lr()[0]:.6f, INFO: {name}')
        
        # continue
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for (inputs, targets) in tqdm(val_loader, ncols=60):
                if inputs.size(0) == 0:
                    print("Skip Zero-Size")
                    continue
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Transformer
                # 다음 단어를 마스킹하려면 마스크 가져오기
                # sequence_length = targets.size(1)
                # tgt_mask = model.get_tgt_mask(sequence_length).to(device)
                # outputs = model(inputs, inputs, tgt_mask)
                
                # LSTM
                # outputs = model(inputs)
                
                # T5
                outputs = model(inputs, labels=targets)
                
                # output_ids = torch.argmax(outputs, dim=2)
                
                #T5
                output_ids = torch.argmax(outputs.logits, dim=2)
                
                # output_ids = torch.argmax(outputs, dim=2)
                # loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                loss = criterion(outputs.logits.view(-1, vocab_size), targets.view(-1))
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Stopping training, encountered NaN/Inf loss at epoch {epoch}")
                    continue
                
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
        val_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}')
        
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        ######## TODO
        for (inputs, targets) in tqdm(test_loader, ncols=60):
            if inputs.size(0) == 0:
                print("Skip Zero-Size")
                continue
            seq_len = inputs.shape[1]

            if seq_len < 20:
                input_len = 10
                max_len = seq_len
            elif seq_len < 100:
                input_len = 20
                max_len = seq_len
            else:
                input_len = 20
                max_len = 100
                
            inputs = inputs[:,:input_len].to(device)
            targets = targets[:,:max_len].to(device)

            # out = model.infer(inputs, length=max_len)
            out = model.transformer.infer(inputs)
            
            out = out[:,1:max_len+1]
            
            total, correct, acc = cal_acc(out, targets)

            total_cnt += total
            total_correct += correct
            total_acc += acc
        
        test_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Test Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}')
            
        if not os.path.exists(f'../../../workspace/out/chord/{name}'):
            os.makedirs(f'../../../workspace/out/chord/{name}')
            
        with open(f'../../../workspace/out/chord/{name}/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(f'../../../workspace/out/chord/{name}/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)
        with open(f'../../../workspace/out/chord/{name}/train_acc.pkl', 'wb') as file:
            pickle.dump(train_acc, file)
        with open(f'../../../workspace/out/chord/{name}/val_acc.pkl', 'wb') as file:
            pickle.dump(val_acc, file)
        with open(f'../../../workspace/out/chord/{name}/test_acc.pkl', 'wb') as file:
            pickle.dump(test_acc, file)

        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            patience_counter = 0

            torch.save(model.state_dict(), f'../../../workspace/out/chord/{name}/model_{epoch+1}_{val_loss[-1]:.6f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]}')        
        else:
            patience_counter += 1
            
        if patience_counter == 5:
            print("Early Stop")
            print("Finish The Train Model")
            # break
      

def cal_acc(infer, target):
    if infer.shape[0] != target.shape[0] or infer.shape[1] != target.shape[1]:
        raise Exception
    
    # if verbose:
    #     if detail:
    #         print("CAL ACC")
    #         print(infer.shape)
    #         print(target.shape)
        
    correct = 0
    total = 0
    
    for i in range(infer.shape[0]):
            
        zero_infer = torch.where(infer[i] == 0)[0]
        zero_target = torch.where(target[i] == 0)[0]
        
        zero_infer = zero_infer[0].item() if len(zero_infer) > 0 else infer[i].shape[0]
        zero_target = zero_target[0].item() if len(zero_target) > 0 else target[i].shape[0]
        
        max_zero = max(zero_infer, zero_target)
        
        # if verbose:
        #     if detail:
        #         # print("IN FOR")
        #         # print(infer[i].shape)
        #         # print(infer[i][:10])
        #         # print(infer[i][-10:])
        #         # print(target[i].shape)
        #         # print(target[i][:10])
        #         # print(target[i][-10:])
        #         print(zero_infer)
        #         print(zero_target)
        #         print(max_zero)
        
        infer_slice = infer[i][:max_zero]
        target_slice = target[i][:max_zero]
        
        correct += (infer_slice == target_slice).sum().item()
        total += max_zero
    if total == 0.0 or total == 0:
        total = 1

    return total, correct, correct/total
        
        
        
        
        
run()