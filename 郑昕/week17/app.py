import numpy as np
import pandas as pd
from collections import defaultdict

def load_data(file_path):
    """
    从Excel文件加载电影评分数据
    :param file_path: 数据文件路径
    :return: 
        user_to_rating: 用户到物品评分的字典 {user_id: [rating1, rating2, ...]}
        item_id_to_name: 物品ID到名称的映射 {item_id: item_name}
    """
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=0)
    
    # 提取电影名称作为物品名称
    item_id_to_name = {i+1: name for i, name in enumerate(df.iloc[:, 1])}
    
    # 假设数据中只有一行用户评分（如示例数据所示）
    # 如果有多个用户，需要调整这部分代码
    ratings = df.iloc[0, 2:].values.astype(float)
    
    # 创建用户到评分的字典（这里假设只有一个用户）
    user_to_rating = {1: ratings}
    
    return user_to_rating, item_id_to_name

class ItemCF:
    def __init__(self, user_to_rating, item_id_to_name):
        """
        初始化ItemCF推荐系统
        :param user_to_rating: 用户到物品评分的字典 {user_id: [rating1, rating2, ...]}
        :param item_id_to_name: 物品ID到名称的映射 {item_id: item_name}
        """
        self.user_to_rating = user_to_rating
        self.item_id_to_name = item_id_to_name
        self.item_similarity = None
        self.item_to_vector = None
        
    def build_item_vectors(self):
        """构建物品向量，每个物品的向量是所有用户对它的评分"""
        self.item_to_vector = {}
        total_users = len(self.user_to_rating)
        
        for user_id, ratings in self.user_to_rating.items():
            for item_id, rating in enumerate(ratings, start=1):
                if item_id not in self.item_to_vector:
                    self.item_to_vector[item_id] = np.zeros(total_users + 1)
                self.item_to_vector[item_id][user_id] = rating
                
    def cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-9)  # 防止除以0
        
    def calculate_similarity(self):
        """计算物品之间的相似度矩阵"""
        if self.item_to_vector is None:
            self.build_item_vectors()
            
        item_ids = list(self.item_to_vector.keys())
        n_items = len(item_ids)
        self.item_similarity = np.zeros((n_items + 1, n_items + 1))  # +1因为物品ID从1开始
        
        for i in item_ids:
            for j in item_ids:
                if i == j:
                    self.item_similarity[i][j] = 1.0
                elif j > i:  # 避免重复计算
                    sim = self.cosine_similarity(self.item_to_vector[i], self.item_to_vector[j])
                    self.item_similarity[i][j] = sim
                    self.item_similarity[j][i] = sim
                    
    def recommend_items(self, user_id, top_n=10):
        """
        为用户推荐物品
        :param user_id: 用户ID
        :param top_n: 推荐物品数量
        :return: 推荐的物品列表 [(item_name, score), ...]
        """
        if self.item_similarity is None:
            self.calculate_similarity()
            
        user_ratings = self.user_to_rating[user_id]
        item_scores = defaultdict(float)
        
        # 遍历用户评分过的物品
        for item_id, rating in enumerate(user_ratings, start=1):
            if rating > 0:  # 用户对该物品有评分
                # 找到与该物品最相似的top_n个物品
                similar_items = np.argsort(self.item_similarity[item_id])[::-1][1:top_n+1]
                
                for sim_item_id in similar_items:
                    if user_ratings[sim_item_id - 1] == 0:  # 用户未评分过的物品
                        # 加权分数 = 相似度 * 用户对原物品的评分
                        item_scores[sim_item_id] += self.item_similarity[item_id][sim_item_id] * rating
        
        # 按分数排序并返回结果
        recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(self.item_id_to_name[item_id], score) for item_id, score in recommended_items]

    def predict_rating(self, user_id, item_id, top_k=10):
        """
        预测用户对某物品的评分
        :param user_id: 用户ID
        :param item_id: 物品ID
        :param top_k: 使用最相似的k个物品
        :return: 预测评分
        """
        if self.item_similarity is None:
            self.calculate_similarity()
            
        user_ratings = self.user_to_rating[user_id]
        rated_items = [i for i, r in enumerate(user_ratings, start=1) if r > 0]
        
        # 找到与目标物品最相似的top_k个已评分物品
        similarities = [(i, self.item_similarity[item_id][i]) for i in rated_items]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_items = similarities[:top_k]
        
        # 计算加权平均评分
        sum_sim = sum(rating * sim for i, sim in top_similar_items for rating in [user_ratings[i-1]])
        sum_weights = sum(sim for _, sim in top_similar_items)
        
        return sum_sim / (sum_weights + 1e-9)  # 防止除以0


# 使用示例
if __name__ == "__main__":
    # 加载数据
    file_path = "user_movie_rating.xlsx"  # 替换为您的实际文件路径
    user_to_rating, item_id_to_name = load_data(file_path)
    
    # 初始化ItemCF
    item_cf = ItemCF(user_to_rating, item_id_to_name)
    
    # 计算物品相似度
    item_cf.calculate_similarity()
    
    # 为用户1推荐10个物品
    recommendations = item_cf.recommend_items(user_id=1, top_n=10)
    print("为用户1推荐的物品:")
    for item, score in recommendations:
        print(f"{item}: {score:.4f}")
    
    # 预测用户1对物品5的评分
    predicted_rating = item_cf.predict_rating(user_id=1, item_id=5)
    print(f"预测用户1对物品5的评分: {predicted_rating:.2f}")