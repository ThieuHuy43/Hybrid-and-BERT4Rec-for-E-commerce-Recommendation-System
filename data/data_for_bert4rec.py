import pandas as pd

df = pd.read_csv("review_df.csv")

# Tạo cột 'timestamp' là thứ tự mua hàng của mỗi author_id
df["timestamp"] = df.groupby("author_id").cumcount() + 1

# Chỉ giữ 3 cột cần thiết
df = df[["author_id", "product_id", "timestamp"]]

df.to_csv("inter.csv", index=False)

print(df.head())
