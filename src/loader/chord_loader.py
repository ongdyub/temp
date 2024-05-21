import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ChordDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]

        return input_ids
    
def create_dataloaders(batch_size):
    chord_data = torch.load('../../../workspace/data/tensor/chord_tensor.pt')
    train, val_test = train_test_split(chord_data, test_size=0.2, random_state=5)
    val, test = train_test_split(val_test, test_size=0.5, random_state=5)
    
    train_dataset = ChordDataset(train)
    val_dataset = ChordDataset(val)
    test_dataset = ChordDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader