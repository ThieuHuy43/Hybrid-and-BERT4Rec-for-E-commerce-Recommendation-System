import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from dataset import ProductDataset, load_data
import pandas as pd
from model import BERT4Rec
from torch.utils.tensorboard import SummaryWriter
import os

def parse_args():
    parser = argparse.ArgumentParser(description="BERT4Rec Training")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")
    
    parser.add_argument("--log_dir", type=str, default="./runs/bert4rec", help="TensorBoard log directory")
    parser.add_argument("--data_path", type=str, default=r"C:\Users\PC\OneDrive\Dokumen\recom\ecom_recommendation_project\data\inter.csv")

    return parser.parse_args()

def split_train_test(df, min_seq_len=5, n_test=1):
    df = df.copy()
    df["order"] = df.groupby("author_id").cumcount()
    df = df.sort_values(by=["author_id", "order"])

    # Chỉ giữ user đủ dài
    user_counts = df["author_id"].value_counts()
    valid_users = user_counts[user_counts >= min_seq_len].index
    df = df[df["author_id"].isin(valid_users)]

    # Lấy n_test item cuối làm test
    test_df = df.groupby("author_id").tail(n_test)
    train_df = df.drop(test_df.index)

    return train_df, test_df

def recall_at_k(preds, target, k):
    top_k = preds.topk(k, dim=1).indices
    correct = top_k.eq(target.view(-1, 1)).sum().item()
    return correct / target.size(0)

def evaluate(model, dataloader, device):
    model.eval()
    recalls = {5: 0, 10: 0, 20: 0}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)

            for k in [5, 10, 20]:
                top_k = outputs.topk(k, dim=1).indices
                match = (top_k == labels.view(-1, 1))
                recalls[k] += match.sum().item()
            total += labels.size(0)

    if total == 0:
        print("Không có dữ liệu để đánh giá! Test set rỗng hoặc không hợp lệ.")
        return {f"Recall@{k}": 0.0 for k in [5, 10, 20]}

    return {f"Recall@{k}": recalls[k] / total for k in recalls}

def train(model, dataloader, optimizer, criterion, device, writer=None, epoch=0):
    model.train()
    total_loss = 0
    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)

    for i, batch in progress:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

        if writer:
            writer.add_scalar("Loss/batch", loss.item(), epoch * len(dataloader) + i)

    return total_loss / len(dataloader)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, user2id, product2id = load_data(args.data_path)
    #df.to_csv("users_interactions.csv", index = False)
    train_df, test_df = split_train_test(df, 5, 2)

    train_dataset = ProductDataset(train_df, max_len=args.max_len, is_train=True)
    test_dataset = ProductDataset(test_df, max_len=args.max_len, is_train=False)

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)

    model = BERT4Rec(vocab_size=len(product2id) + 1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    print("Model is on:", next(model.parameters()).device)

    for epoch in range(1, args.epochs + 1):
        print(f"\n\033[96m[Epoch {epoch}/{args.epochs}]\033[0m Starting training...")

        loss = train(model, train_loader, optimizer, criterion, device, writer=writer, epoch=epoch)

        recalls = evaluate(model, test_loader, device)

        print(f"\033[92mEpoch {epoch} - Loss: {loss:.4f}\033[0m")
        for k, v in recalls.items():
            print(f"{k}: \033[93m{v:.4f}\033[0m")
            writer.add_scalar(k, v, epoch)
        writer.add_scalar("Loss/train", loss, epoch)
        for k in recalls:
            writer.add_scalar(f"{k}", recalls[k], epoch)

    writer.close()
    torch.save(model.state_dict(), "best_bert4rec_model.pth")

if __name__ == "__main__":
    main()
