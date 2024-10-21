from tqdm import tqdm
import json
import torch
import torch
import torch.nn.functional as F

target_rows = 2050

def pad_tensor_to_target(tensor, target_size):
    num_rows_to_pad = target_size[0] - tensor.size(0)
    padded_tensor = F.pad(tensor, (0, 0, 0, num_rows_to_pad), "constant", 0)
    return padded_tensor

raw_data_path = '../../data/corpus/raw_corpus_bpe.txt'
# raw_data_path = '../../data/corpus/first_5_lines_bpe.txt'
output_dir = '../../data/corpus/'

vocab_path = '../vocabs/inst.json'
with open(vocab_path, 'r') as file:
    vocab = json.load(file)

raw_data = []
with open(raw_data_path, 'r') as f:
    for line in tqdm(f, desc="reading original txt file..."):
        raw_data.append(line.strip())

inst_data = torch.zeros((1,2050,32), dtype=torch.int16)

inst_cnt = 0
# max_cnt = -1
# length = []
part = 50000
cnt = 0
for song in tqdm(raw_data, desc="process each txt file..."):
    if cnt < 40000:
        cnt += 1
        continue
    
    if cnt == part:
        break
    
    token_list = song.split(' ')
    
    start = torch.ones((1,32), dtype=torch.int16)

    inst_list = []
    inst_tensor = torch.ones((1,32), dtype=torch.int16)
    inst_tensor *= 2
    
    for idx, token in enumerate(token_list):
        
        if token[0] == 'm' or token[0] == 'M':
            if len(inst_list) > 0:
                inst_tensor[0, :len(inst_list)] = torch.tensor(inst_list, dtype=torch.int16)
            
            start = torch.cat((start, inst_tensor), dim=0)
            
            inst_tensor = torch.ones((1,32), dtype=torch.int16)
            inst_list = []
        
        if token[0] == 'x' or token[0] == 'X' or token[0] == 'y':
            inst_idx = vocab[token]
            
            if inst_idx in inst_list:
                pass
            else:
                inst_list.append(inst_idx)
        
    if len(inst_list) > 0:
        inst_tensor[0, :len(inst_list)] = torch.tensor(inst_list, dtype=torch.int16)   
    start = torch.cat((start, inst_tensor), dim=0)
    
    eos_tensor = torch.zeros((1,32), dtype=torch.int16)
    start = torch.cat((start, eos_tensor), dim=0)
    
    start = pad_tensor_to_target(start, (target_rows, 32))
    start = start.unsqueeze(0)
    inst_data = torch.cat((inst_data, start), dim=0)
    
    cnt += 1
    # print(max_cnt)
    # length.append(max_cnt)

# print(len(length))
# print(max(length))
torch.save(inst_data[1:,1:-1,:], f'./inst_tensor_{part}.pt')