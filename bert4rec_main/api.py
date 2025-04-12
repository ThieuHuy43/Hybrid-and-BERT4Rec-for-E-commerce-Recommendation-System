from flask import Flask, request, jsonify
import pandas as pd
import pickle
import torch

# ---------- Load mô hình hybrid similarity ----------
hybrid_model_path = r"C:\Users\PC\OneDrive\Dokumen\recom\ecom_recommendation_project\hybrid_rec\hybrid_similarity.pkl"
with open(hybrid_model_path, "rb") as file:
    hybrid_sim = pickle.load(file)

# ---------- Load product info ----------
products_df = pd.read_csv(r"C:\Users\PC\OneDrive\Dokumen\recom\ecom_recommendation_project\data\product_info.csv")
interactions_df = pd.read_csv(r"C:\Users\PC\OneDrive\Dokumen\recom\ecom_recommendation_project\data\inter.csv")

# ---------- Load data, user2id, product2id ----------
from model import BERT4Rec
from dataset import load_data

# Load lại dữ liệu tương tác
df, user2id, product2id = load_data(r"C:\Users\PC\OneDrive\Dokumen\recom\ecom_recommendation_project\data\inter.csv")
id2product = {v: k for k, v in product2id.items()}

# ---------- Load mô hình BERT4Rec đã huấn luyện ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT4Rec(vocab_size=len(product2id)+1).to(device)

checkpoint = torch.load("best_bert4rec_model.pth", map_location=device)
model_state_dict = model.state_dict()

matched = {}
skipped = {}

# Phân loại từng layer
for k, v in checkpoint.items():
    if k in model_state_dict:
        if v.shape == model_state_dict[k].shape:
            matched[k] = v
        else:
            skipped[k] = f"Shape mismatch: checkpoint {v.shape} vs model {model_state_dict[k].shape}"
    else:
        skipped[k] = "Not found in model"

print(f"\nMatched layers: {len(matched)}")
print(f"Skipped layers: {len(skipped)}")

if skipped:
    print("Chi tiết các layer bị bỏ qua:")
    for name, reason in skipped.items():
        print(f" - {name}: {reason}")

model_state_dict.update(matched)
model.load_state_dict(model_state_dict)
#model.load_state_dict(torch.load(bert_path, map_location=device))
model.eval()

# ---------- Flask App ----------
app = Flask(__name__)

# ------------- Hybrid Recommend theo sản phẩm -------------
def recommend_hybrid(product_id, num_recommendations=5):
    product_id = str(product_id)
    if product_id not in hybrid_sim.index:
        return []

    scores = hybrid_sim.loc[product_id].sort_values(ascending=False)
    recommended_products = scores.iloc[1:num_recommendations+1].index.astype(str)

    return products_df[products_df["product_id"].isin(recommended_products)][
        ["product_name", "brand_name","loves_count", "rating", "price_usd"]
    ].to_dict(orient="records")

# ------------- BERT4Rec Recommend theo user -------------
def recommend_bert4rec(author_id, topk=5, max_len=50):
    if author_id not in user2id:
        return []

    uid = user2id[author_id]
    
    # Kiểm tra cột "order" có tồn tại không
    if "order" not in df.columns:
        df["order"] = df.groupby("author_id").cumcount() + 1

    user_df = df[df["author_id"] == uid].sort_values(by="order")

    if len(user_df) == 0:
        return []

    input_seq = user_df["product_id"].tolist()[-max_len:]
    pad_len = max_len - len(input_seq)
    input_seq = [0] * pad_len + input_seq

    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=-1)
        topk_indices = torch.topk(probs, topk).indices[0].cpu().tolist()

    product_ids = [id2product.get(idx) for idx in topk_indices if idx in id2product]

    recommendations = products_df[products_df["product_id"].isin(product_ids)][
        ["product_name", "brand_name","loves_count", "rating", "price_usd"]
    ].drop_duplicates()

    return recommendations.to_dict(orient="records")

# ------------- API Routes -------------
@app.route("/recommend/hybrid", methods=["GET"])
def recommend_hybrid_route():
    product_id = request.args.get("product_id")
    num = request.args.get("num_recommendations", default=5, type=int)

    if not product_id:
        return jsonify({"error": "Missing product_id"}), 400

    recs = recommend_hybrid(product_id, num)
    if not recs:
        return jsonify({"error": "No hybrid recommendations found"}), 404

    return jsonify({"recommendations": recs})


@app.route("/recommend/bert4rec", methods=["GET"])
def recommend_bert_route():
    author_id = request.args.get("author_id")
    topk = request.args.get("topk", default=5, type=int)

    if not author_id:
        return jsonify({"error": "Missing author_id"}), 400

    recs = recommend_bert4rec(author_id, topk)
    if not recs:
        return jsonify({"error": "No BERT4Rec recommendations found or invalid user ID."}), 404

    return jsonify({"recommendations": recs})


@app.route("/users", methods=["GET"])
def get_all_users():
    return jsonify({"author_ids": list(user2id.keys())})

@app.route("/user_interactions", methods=["GET"])
def get_user_interactions():
    author_id = request.args.get("author_id")
    try:
        author_id = str(author_id)
    except:
        return jsonify([])

    user_history = interactions_df[interactions_df["author_id"] == author_id]

    if user_history.empty:
        return jsonify([])

    user_history = user_history.merge(products_df, on="product_id", how="left")
    result = user_history[["product_id", "product_name", "timestamp"]]\
                         .sort_values(by="timestamp", ascending=False)
    return jsonify(result.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)

