import streamlit as st
import pandas as pd
import requests

# ---------- Giao diện chính ----------
st.set_page_config(page_title="E-Commerce Recommender", layout="wide")
st.title("🛒 E-Commerce Recommender System")
st.markdown("Khám phá hệ thống gợi ý sản phẩm bằng mô hình Hybrid và BERT4Rec.")

# ---------- Config ----------
API_URL = "http://127.0.0.1:5001"
PRODUCT_CSV = r"C:\Users\PC\OneDrive\Dokumen\recom\ecom_recommendation_project\data\products.csv"

# ---------- Load dữ liệu sản phẩm ----------
@st.cache_data
def load_products():
    return pd.read_csv(PRODUCT_CSV)

products_df = load_products()
categories = products_df["primary_category"].dropna().unique()
tab1, tab2 = st.tabs(["📦 Hybrid Recommender", "🧠 BERT4Rec Recommender"])

# ---------- Tab 1: Hybrid Recommender ----------
with tab1:
    st.subheader("📦 Hybrid Recommendation theo Sản phẩm")
    st.markdown("Chọn một danh mục và sản phẩm cụ thể để xem các sản phẩm tương tự được gợi ý.")

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_category = st.selectbox("🔎 Chọn danh mục sản phẩm:", sorted(categories))

    filtered_products = products_df[products_df["primary_category"] == selected_category]

    with col2:
        product_options = filtered_products.apply(lambda x: f"{x['product_name']} (ID: {x['product_id']})", axis=1)
        selected_product_label = st.selectbox("🎯 Chọn sản phẩm:", product_options)
        product_id = selected_product_label.split("ID: ")[-1].replace(")", "")

    num_recs = st.slider("📌 Số lượng gợi ý:", min_value=1, max_value=10, value=5)

    if st.button("📥 Lấy gợi ý từ Hybrid Model"):
        try:
            params = {"product_id": product_id, "num_recommendations": num_recs}
            response = requests.get(f"{API_URL}/recommend/hybrid", params=params)
            response.raise_for_status()
            recs = response.json()["recommendations"]

            if recs:
                st.success("✅ Gợi ý sản phẩm thành công!")
                st.dataframe(pd.DataFrame(recs), use_container_width=True)
            else:
                st.warning("Không tìm thấy gợi ý phù hợp.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi gọi API: {e}")

# ---------- Tab 2: BERT4Rec Recommender ----------
with tab2:
    st.subheader("👤 Gợi ý theo Người dùng với BERT4Rec")
    st.markdown("Chọn một người dùng và số lượng sản phẩm muốn gợi ý.")

    try:
        users_response = requests.get(f"{API_URL}/users")
        users_response.raise_for_status()
        author_ids = users_response.json().get("author_ids", [])
    except Exception as e:
        author_ids = []
        st.error(f"Không thể tải danh sách người dùng từ API: {e}")

    if author_ids:
        col1, col2 = st.columns(2)
        with col1:
            sorted_author_ids = sorted(author_ids)[:100]  # Giới hạn chỉ lấy 100 người đầu để gợi ý
            selected_user = st.selectbox("🙍‍♂️ Chọn người dùng:", sorted(sorted_author_ids))
        with col2:
            topk = st.slider("📌 Số sản phẩm muốn gợi ý:", min_value=1, max_value=20, value=5)

        # Hiển thị các sản phẩm đã tương tác
        try:
            history_params = {"author_id": selected_user}
            history_response = requests.get(f"{API_URL}/user_interactions", params=history_params)
            history_response.raise_for_status()
            history = history_response.json()

            if history:
                st.markdown("🕘 **Lịch sử tương tác của người dùng**")
                st.dataframe(pd.DataFrame(history), use_container_width=True)
            # else:
            #     st.info("ℹ️ Người dùng này chưa có tương tác nào.")
        except Exception as e:
            st.error(f"🚨 Không thể tải lịch sử tương tác: {e}")

        if st.button("📥 Lấy gợi ý từ BERT4Rec"):
            try:
                params = {"author_id": selected_user, "topk": topk}
                response = requests.get(f"{API_URL}/recommend/bert4rec", params=params)
                response.raise_for_status()
                recs = response.json()["recommendations"]

                if recs:
                    st.success("Gợi ý sản phẩm thành công!")
                    st.dataframe(pd.DataFrame(recs), use_container_width=True)
                else:
                    st.warning("Người dùng này chưa có lịch sử mua hàng đủ để gợi ý.")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi gọi API: {e}")

    else:
        st.warning("Không có người dùng nào để gợi ý. Hãy đảm bảo API đang chạy và dữ liệu được load.")
