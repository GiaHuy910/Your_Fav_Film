import pandas as pd
import sys
import ast
sys.stdout.reconfigure(encoding='utf-8')

# Đọc dữ liệu
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

mcredit=pd.merge(movies,credits,left_on="id",right_on="movie_id")

def Convert(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
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
    
mcredit['genres']=mcredit['genres'].apply(Convert)
mcredit['keywords']=mcredit['keywords'].apply(Convert)
mcredit['cast']=mcredit['cast'].apply(Convert)
mcredit['crew']=mcredit['crew'].apply(Convert)
mcredit['overview']=mcredit['overview'].fillna("").apply(lambda x: x.split())

mcredit['title'] = mcredit['original_title']
print(mcredit[['title','genres', 'cast', 'crew', 'keywords','overview']].head())
