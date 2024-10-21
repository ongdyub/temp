from tqdm import tqdm
import json
import torch

raw_data_path = '../../data/corpus/raw_corpus_bpe.txt'
# raw_data_path = '../../data/corpus/first_5_lines_bpe.txt'
output_dir = '../../data/corpus/'

vocab_path = '../vocabs/chord.json'
with open(vocab_path, 'r') as file:
    vocab = json.load(file)

raw_data = []
with open(raw_data_path, 'r') as f:
    for line in tqdm(f, desc="reading original txt file..."):
        raw_data.append(line.strip())

bos = vocab['<bos>']
pad = vocab['<pad>']
eos = vocab['<eos>']
# print(len(raw_data))
# print(raw_data[0][:20])

# length = []
chord_data = torch.empty((1,2048), dtype=torch.int16)

for song in tqdm(raw_data, desc="process each txt file..."):
    token_list = song.split(' ')
    # print(token_list[:10])
    cnt = 0
    chord_seq = []
    chord_tensor = torch.zeros((1,2048), dtype=torch.int16)
    chord_tensor[0,0] = torch.tensor(bos, dtype=torch.int16)
    for token in token_list:
        if token[0] == 'h' or token[0] == 'H':
        #    print(token)
        #    print(vocab[token])
           chord_seq.append(vocab[token])
    # print(cnt)
    chord_tensor[0, 1:len(chord_seq)+1] = torch.tensor(chord_seq, dtype=torch.int16)
    # length.append(cnt)
    # print(chord_tensor)
    chord_data = torch.cat((chord_data, chord_tensor), dim=0)
    
# print(len(length))
# print(max(length))
# print(chord_data[1:])
print(chord_data[1:].shape)

torch.save(chord_data[1:], './chord_tensor.pt')