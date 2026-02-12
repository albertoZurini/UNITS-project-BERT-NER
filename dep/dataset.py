import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn

class MyDataset(Dataset):
    """
    Class to store the tweet data as a PyTorch Dataset.
    """
    def __init__(self, encodings, attention_masks, labels):
        self.encodings = encodings
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings[idx]),
            "attention_mask": torch.tensor(self.attention_masks[idx]),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

def collate(batch):
    ips = [item['input_ids'][0] for item in batch]
    attn = [item['attention_mask'][0] for item in batch]
    lb = [item['labels'][0] for item in batch]

    # Pad sequences to the same length.
    ips_padded = torch.nn.utils.rnn.pad_sequence(ips, batch_first=True, padding_value=0)
    attn_padded = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    lb_padded = torch.nn.utils.rnn.pad_sequence(lb, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': ips_padded,
        'attention_mask': attn_padded,
        'labels': lb_padded
    }
