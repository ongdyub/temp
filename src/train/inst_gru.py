import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..model.inst_model import InstGRU, InstPooling, InstSimplePooling
from ..model.inst_transformer import InstTransformer
from ..loader.inst_loader import InstGRUDataset, create_InstGRU
import setproctitle

verbose = False
detail = False

device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 1
num_epochs = 100
#############################################################
#############################################################
#############################################################


#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
folder = 'inst_gru'
for i in range(1):
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    base_model = 'GRU'
    model = InstGRU()
    name = f'{base_model}_Vanila'
    
    base_model = 'POOL'
    model = InstPooling()
    name = f'{base_model}_Vanila'
    
    base_model = 'POOL'
    model = InstSimplePooling()
    name = f'{base_model}_Simple_Vanila'
    
    base_model = 'TF'
    model = InstTransformer(device=device, num_heads=8, d_ff=2048, num_layers=6)
    name = f'{base_model}_Vanila'


for i in range(1):
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    criterion = nn.BCEWithLogitsLoss()
    accuracy_threshold = 0.95
    gru_frozen = [False] * 133

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


train_loader, val_loader, test_loader = create_InstGRU(batch_size)
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
    
    if type == 'GRU':
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
        
            total_cnt = 0
            total_correct = 0
            total_acc = 0
            
            for chord_tensor, init_inst, targets in tqdm(train_loader, ncols=60):
                chord_tensor = chord_tensor.to(device)
                init_inst = init_inst.to(device)
                # print(targets)
                targets = [t.to(device) for t in targets[0]]
                
                optimizer.zero_grad()
                # print("AAA")
                # print(chord_tensor)
                # print(chord_tensor.shape)
                # print(init_inst)
                outputs = model(chord_tensor, init_inst)
                
                for i in range(133):
                    if epoch > 50 and gru_frozen[i]:
                        continue
                    # print("DDDDD")
                    output = outputs[i]
                    target = targets[i]
                    
                    output = output.view(-1)
                    target = target.view(-1)
                    
                    # print(output)
                    # print(target)
                    loss = criterion(output, target.float())
                    loss.backward(retain_graph=True)
                    accuracy = calculate_accuracy(output, target)
                    if epoch > 50 and accuracy > accuracy_threshold:
                        print(f'Frozen {i} {accuracy}')
                        gru_frozen[i] = True
                        
                    # if i==132:
                    #     print("132 INST")
                    #     print(loss)
                    #     print(accuracy)
                    #     print((torch.sigmoid(output) > 0.5).float())
                    #     print(target)
                        
                    total_train_loss += loss.item()
                    total_acc += accuracy
                    total_cnt += 1
                    
                optimizer.step()
                # total_train_loss /= (133*len(train_loader))
                # total_acc /= (133*len(train_loader))
                # total_acc /= (133)
                # break
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss / total_cnt:.8f} Acc: {total_acc / total_cnt:.8f}')
            print(f'GRUs frozen: {sum(gru_frozen)} / 133')
                    
            
        pass
    elif type == 'POOL':
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            total_cnt = 0
            total_acc = 0
            
            for chord_tensor, init_inst, targets in tqdm(train_loader, ncols=60):
                chord_tensor = chord_tensor.to(device)
                init_inst = init_inst.to(device)

                targets = [t.to(device) for t in targets[0]]
                
                optimizer.zero_grad()
                outputs = model(chord_tensor, init_inst)
                
                for i in range(133):
                    if epoch > 50 and gru_frozen[i]:
                        continue
                    
                    output = outputs[i]
                    target = targets[i]
                    
                    output = output.view(-1)
                    target = target.view(-1)
                    
                    loss = criterion(output, target.float())
                    loss.backward(retain_graph=True)
                    accuracy = calculate_accuracy(output, target)
                    if epoch > 50 and accuracy > accuracy_threshold:
                        print(f'Frozen {i} {accuracy}')
                        gru_frozen[i] = True
                        
                    total_train_loss += loss.item()
                    total_acc += accuracy
                    total_cnt += 1
                    
                optimizer.step()  
  
            scheduler.step()
            
            train_loss.append(total_train_loss / total_cnt)
            train_acc.append(total_acc / total_cnt)
            print(f'Linears frozen: {sum(gru_frozen)} / 133')
            print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.8f}, Acc: {train_acc[-1]:.8f} LR: {scheduler.get_last_lr()[0]:.8f}, INFO: {name}')
            
            
            model.eval()
            total_val_loss = 0
            total_cnt = 0
            total_acc = 0
            
            with torch.no_grad():
                for chord_tensor, init_inst, targets in tqdm(val_loader, ncols=60):
                    chord_tensor = chord_tensor.to(device)
                    init_inst = init_inst.to(device)

                    targets = [t.to(device) for t in targets[0]]
                    optimizer.zero_grad()
                    outputs = model(chord_tensor, init_inst)
                    
                    for i in range(133):
                        output = outputs[i]
                        target = targets[i]
                        
                        output = output.view(-1)
                        target = target.view(-1)

                        loss = criterion(output, target.float())
                        
                        accuracy = calculate_accuracy(output, target)
                            
                        total_val_loss += loss.item()
                        total_acc += accuracy
                        total_cnt += 1
                    # break
                    
            val_loss.append(total_val_loss / total_cnt)
            val_acc.append(total_acc / total_cnt)
            print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.8f}, Acc: {val_acc[-1]:.8f}, INFO: {name}')

            total_cnt = 0
            total_acc = 0
            
            with torch.no_grad():
                for chord_tensor, init_inst, targets in tqdm(test_loader, ncols=60):
                    chord_tensor = chord_tensor.to(device)
                    init_inst = init_inst.to(device)

                    targets = [t.to(device) for t in targets[0]]
                    optimizer.zero_grad()
                    outputs = model(chord_tensor, init_inst)
                    
                    for i in range(133):
                        
                        if init_inst[:,i] == 0:
                            continue
                        output = outputs[i]
                        target = targets[i]
                        
                        output = output.view(-1)
                        target = target.view(-1)
                        
                        accuracy = calculate_accuracy(output, target)

                        total_acc += accuracy
                        total_cnt += 1
                    # break
                    
            test_acc.append(total_acc / total_cnt)
            print(f'Epoch {epoch+1}, Acc: {test_acc[-1]:.8f}, INFO: {name}')
            
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
                
    elif type == 'TF':
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            total_cnt = 0
            total_acc = 0
            
            for chord_tensor, init_inst, targets in tqdm(train_loader, ncols=60):
                chord_tensor = chord_tensor.to(device)
                init_inst = init_inst.to(device)

                targets = [t.to(device) for t in targets[0]]
                
                optimizer.zero_grad()
                
                # print("TRAIN")
                # print(chord_tensor.shape)
                # print(init_inst.shape)
                
                
                for i in range(133):
                    if init_inst[:,i] == 0:
                            continue
                    optimizer.zero_grad()
                    outputs = model(chord_tensor, init_inst, i)
                    output = outputs
                    target = targets[i]
                    # print(output.shape)
                    # print(target.shape)
                    output = output.view(-1)
                    target = target.view(-1)
                    # print("INDEICD")
                    # print(output.shape)
                    # print(target.shape)
                    loss = criterion(output, target.float())
                    loss.backward()
                    accuracy = calculate_accuracy(output, target)
                        
                    total_train_loss += loss.item()
                    total_acc += accuracy
                    total_cnt += 1
                    
                    optimizer.step()
                    
                # break
  
            scheduler.step()
            
            train_loss.append(total_train_loss / total_cnt)
            train_acc.append(total_acc / total_cnt)
            # print(f'Linears frozen: {sum(gru_frozen)} / 133')
            print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.8f}, Acc: {train_acc[-1]:.8f} LR: {scheduler.get_last_lr()[0]:.8f}, INFO: {name}')
            
            
            model.eval()
            total_val_loss = 0
            total_cnt = 0
            total_acc = 0
            
            with torch.no_grad():
                for chord_tensor, init_inst, targets in tqdm(val_loader, ncols=60):
                    chord_tensor = chord_tensor.to(device)
                    init_inst = init_inst.to(device)

                    targets = [t.to(device) for t in targets[0]]
                    
                    for i in range(133):
                        if init_inst[:,i] == 0:
                            continue
                        outputs = model(chord_tensor, init_inst, i)
                        output = outputs
                        target = targets[i]
                        
                        output = output.view(-1)
                        target = target.view(-1)

                        loss = criterion(output, target.float())
                        
                        accuracy = calculate_accuracy(output, target)
                            
                        total_val_loss += loss.item()
                        total_acc += accuracy
                        total_cnt += 1
                    # break
                    
            val_loss.append(total_val_loss / total_cnt)
            val_acc.append(total_acc / total_cnt)
            print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.8f}, Acc: {val_acc[-1]:.8f}, INFO: {name}')

            total_cnt = 0
            total_acc = 0
            
            with torch.no_grad():
                for chord_tensor, init_inst, targets in tqdm(test_loader, ncols=60):
                    chord_tensor = chord_tensor.to(device)
                    init_inst = init_inst.to(device)

                    targets = [t.to(device) for t in targets[0]]
    
                    for i in range(133):
                        
                        if init_inst[:,i] == 0:
                            continue
                        
                        outputs = model(chord_tensor, init_inst, i)
                        output = outputs
                        
                        target = targets[i]
                        
                        output = output.view(-1)
                        target = target.view(-1)
                        
                        accuracy = calculate_accuracy(output, target)

                        total_acc += accuracy
                        total_cnt += 1
                    # break
                    
            test_acc.append(total_acc / total_cnt)
            print(f'Epoch {epoch+1}, Acc: {test_acc[-1]:.8f}, INFO: {name}')
            
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
# Function to calculate accuracy
def calculate_accuracy(output, target):
    output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
    predictions = (output > 0.5).float()  # Convert probabilities to binary (0 or 1)
    correct = (predictions == target).float().sum()
    accuracy = correct / target.numel()
    return accuracy.item()

# def save_epoch_loss(epoch, epoch_loss):
#     if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
#         os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
#     with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
#             pickle.dump(epoch_loss, file)

run(base_model)
