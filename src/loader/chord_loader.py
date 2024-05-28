import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class ChordDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        vocab_path = '/workspace/data/vocabs/chord.json'
        vocab_path = '/workspace/data/vocabs/chord_3.json'
        vocab_path = '/workspace/data/vocabs/chord_2.json'
        with open(vocab_path, 'r') as file:
            self.vocab = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_seq = self.data[idx]
        
        if isinstance(text_seq, str):
            toks = text_seq.split()
        l_toks = len(toks)
        ratio = 4
        chord_list = []
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            if t1[0] == 'h' or t1[0] == 'H':
                chord_list.append(t1)
                
        target_chord_seq = self.get_chord_seq(chord_list)
            
        target_chord_tensor = [self.vocab[chd] for chd in target_chord_seq]
        target_chord_tensor = [2] + target_chord_tensor[:766] + [0]
        target_chord_tensor = torch.tensor(target_chord_tensor)

        return target_chord_tensor
    
    # Util functions
    # 3-Group
    # def get_chord_seq(self, chord_list):
    #     group_list = []
    #     for idx in range(2, len(chord_list)):
    #         group_list.append(chord_list[idx-2]+chord_list[idx-1]+chord_list[idx])
            
    #     return group_list
    
    # 2-Group
    def get_chord_seq(self, chord_list):
        group_list = []
        for idx in range(1, len(chord_list)):
            group_list.append(chord_list[idx-1]+chord_list[idx])

        return group_list
    
    # def get_chord_seq(self, chord_list):
    #     group_list = []
    #     for idx in range(0, len(chord_list)):
    #         group_list.append(chord_list[idx])
            
    #     return group_list
    
def create_dataloaders(batch_size):
    # chord_data = torch.load('../../../workspace/data/tensor/chord_tensor.pt')
    raw_data_path = '../../../workspace/data/corpus/raw_corpus_bpe.txt'
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc="reading original txt file..."):
            raw_data.append(line.strip())
            
    # train, val_test = train_test_split(chord_data, test_size=0.1, random_state=5)
    train, val_test = train_test_split(raw_data, test_size=0.1, random_state=5)
    val, test = train_test_split(val_test, test_size=0.2, random_state=5)
    
    train_dataset = ChordDataset(train)
    val_dataset = ChordDataset(val)
    test_dataset = ChordDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, drop_last=True)

    return train_loader, val_loader, test_loader

def collate_batch(batch):
    # padding_value = <eos>
    padded = pad_sequence(batch, padding_value=0, batch_first=True)
    # print("INSIDE BATCH")
    # print(padded.shape)
    # return padded[:,:-1], padded[:,1:]
    return padded[:,:], padded[:,:]