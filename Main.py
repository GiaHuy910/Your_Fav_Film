import pandas as pd
import sys
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
import os
import pickle

# ===========================
# 1. Đọc dữ liệu
# ===========================
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')
mcredit = pd.merge(movies, credits, left_on="id", right_on="movie_id")

def Convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def get_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name']
        return ""
    except:
        return ""

mcredit['genres'] = mcredit['genres'].apply(Convert)
mcredit['keywords'] = mcredit['keywords'].apply(Convert)
mcredit['cast'] = mcredit['cast'].apply(Convert)
mcredit['crew'] = mcredit['crew'].apply(Convert)
mcredit['overview'] = mcredit['overview'].fillna("").apply(lambda x: x.split())
mcredit['title'] = mcredit['original_title']

# ===========================
# 2. Chuẩn bị text kết hợp
# ===========================
def combine_text(row):
    return f"{' '.join(row['genres'])} {' '.join(row['keywords'])} {' '.join(row['cast'])} {' '.join(row['crew'])} {' '.join(row['overview'])}"

mcredit['combined_features'] = mcredit.apply(combine_text, axis=1)

# ===========================
# 3. Tạo hoặc tải Cache Embedding
# ===========================
CACHE_EMB_FILE = "embeddings_cache.pkl"
CACHE_TEXT_FILE = "combined_features_cache.pkl"

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

if os.path.exists(CACHE_EMB_FILE) and os.path.exists(CACHE_TEXT_FILE):
    print("🔹 Tải embeddings từ cache...")
    with open(CACHE_EMB_FILE, "rb") as f:
        embeddings = pickle.load(f)
    with open(CACHE_TEXT_FILE, "rb") as f:
        cached_texts = pickle.load(f)

    # Kiểm tra nếu dữ liệu không thay đổi
    if list(mcredit['combined_features']) != cached_texts:
        print("⚠️ Dataset thay đổi! Tạo lại embeddings...")
        embeddings = model.encode(mcredit['combined_features'], show_progress_bar=True)
        with open(CACHE_EMB_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        with open(CACHE_TEXT_FILE, "wb") as f:
            pickle.dump(list(mcredit['combined_features']), f)
else:
    print("🛠 Lần đầu chạy - tạo embeddings và lưu cache...")
    embeddings = model.encode(mcredit['combined_features'], show_progress_bar=True)
    with open(CACHE_EMB_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    with open(CACHE_TEXT_FILE, "wb") as f:
        pickle.dump(list(mcredit['combined_features']), f)

print("✅ Embeddings đã sẵn sàng!")

# ===========================
# 4. Hàm recommend
# ===========================
def recommend_movie(user_description, top_n=5):
    user_vec = model.encode([user_description])
    sim_scores = cosine_similarity(user_vec, embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    
    print("\n🔍 Gợi ý phim phù hợp với mô tả:")
    for idx in top_indices:
        print(f"- 🎬 {mcredit['title'].iloc[idx]}  (Similarity: {sim_scores[idx]:.4f})")
    
    return mcredit['title'].iloc[top_indices].values

# ===========================
# 5. Test
# ===========================
while True:
    user_input=str(input('Nhập mô tả của bạn(nếu muốn dừng, nhập "end") : '))
    if(user_input=="end"):
        break
    try:
        top_n=int(input('Bạn muốn gợi ý bao nhiêu bộ phim(int) :'))
        recommend_movie(user_input,top_n)
    except:
        raise ValueError('Hãy nhập đúng số nguyên!')



#user_input_vi = "một bộ phim hành động có yếu tố khoa học viễn tưởng và cuộc chiến chống quái vật"
#recommend_movie(user_input_vi, top_n=5)

#user_input_en = "a sci-fi movie with time travel and a hero saving humanity"
#recommend_movie(user_input_en, top_n=5)
