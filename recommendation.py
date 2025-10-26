from fastapi import FastAPI
from pydantic import BaseModel
from Main import recommend_movie, mcredit, model, embeddings
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo FastAPI
app = FastAPI(title="Movie Recommendation API", description="AI gợi ý phim theo mô tả", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Cho tất cả nguồn (hoặc chỉ định http://localhost)
    allow_credentials=True,
    allow_methods=["*"],     # Cho phép tất cả phương thức: POST, GET, OPTIONS
    allow_headers=["*"],     # Cho tất cả header
)
# Định nghĩa cấu trúc dữ liệu nhận từ frontend
class UserRequest(BaseModel):
    description: str
    top_n: int 

@app.post("/recommend")
async def recommend_api(request: UserRequest):
    results = recommend_movie(request.description, request.top_n)
    return {"recommendations": results.tolist()}

@app.get("/")
async def root():
    return {"message": "Welcome to Movie Recommendation API. Use POST /recommend to get suggestions. 2nd modded 3rd modded"}
