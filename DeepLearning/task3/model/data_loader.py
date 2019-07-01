from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, sent, corp):
        self.corp = corp
        self.sent = [self.corp.transform_text(s) for s in sent]

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, index):
        return self.corp.mask_text(self.sent[index])
    
def fetch_dataloader(sent, corp):
    dataset = MyDataset(sent, corp)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    return dataloader