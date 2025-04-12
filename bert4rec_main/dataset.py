import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class ProductDataset(Dataset):
    def __init__(self, dataframe, max_len=50, is_train=True):
        self.data = []
        self.max_len = max_len
        self.is_train = is_train

        for _, group in dataframe.groupby("author_id"):
            product_ids = group.sort_values("order")["product_id"].tolist()

            if is_train:
                for i in range(1, len(product_ids)):
                    input_seq = product_ids[:i][-max_len:]
                    target = product_ids[i]
                    self.data.append((input_seq, target))
            else:
                input_seq = product_ids[:-1][-max_len:]
                target = product_ids[-1]
                self.data.append((input_seq, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        padding = [0] * (self.max_len - len(input_seq))
        input_ids = torch.tensor(padding + input_seq, dtype=torch.long)
        label = torch.tensor(target, dtype=torch.long)
        return {"input_ids": input_ids, "label": label}

def load_data(filepath):
    df = pd.read_csv(filepath)
    user2id = {u: i+1 for i, u in enumerate(df["author_id"].unique())}
    product2id = {p: i+1 for i, p in enumerate(df["product_id"].unique())}

    df["author_id"] = df["author_id"].map(user2id)
    df["product_id"] = df["product_id"].map(product2id)

    df = df.dropna(subset=["author_id", "product_id"])
    df = df.sort_values(by=["author_id", "timestamp"]).reset_index(drop=True)
    df["order"] = df.groupby("author_id").cumcount()

    return df, user2id, product2id

