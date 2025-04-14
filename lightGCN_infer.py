import torch
import pandas as pd
import numpy as np
from torch_geometric.nn import LightGCN
from sklearn.preprocessing import LabelEncoder

# === Load Data and Rebuild Edge Index ===
def load_data():
    train_df = pd.read_csv("data/All_Beauty.train.csv.gz")[['user_id', 'parent_asin', 'rating']]
    train_df.columns = ['userID', 'itemID', 'rating']

    val_df = pd.read_csv("data/All_Beauty.valid.csv.gz")[['user_id', 'parent_asin', 'rating']]
    val_df.columns = ['userID', 'itemID', 'rating']

    test_df = pd.read_csv("data/All_Beauty.test.csv.gz")[['user_id', 'parent_asin', 'rating']]
    test_df.columns = ['userID', 'itemID', 'rating']

    combined_df = pd.concat([train_df, val_df, test_df])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(combined_df['userID'])
    print(combined_df['userID'])
    item_encoder.fit(combined_df['itemID'])

    train_df['userID'] = user_encoder.transform(train_df['userID'])
    train_df['itemID'] = item_encoder.transform(train_df['itemID'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    edge_index = torch.tensor([train_df['userID'].values, train_df['itemID'].values], dtype=torch.long)

    return edge_index, user_encoder, item_encoder, num_users, num_items

# === Load Model from File ===
def load_model(checkpoint_path, num_nodes):
    embedding_dim = 128
    num_layers = 2

    model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model

# === Recommend Top-K Items ===
def recommend(user_raw_id, model, edge_index, user_encoder, item_encoder, num_users, num_items, top_k=10):
    if user_raw_id not in user_encoder.classes_:
        print(f"[WARN] User ID {user_raw_id} not found in training data.")
        print(user_encoder.classes_)
        return []

    user_id = user_encoder.transform([user_raw_id])[0]

    with torch.no_grad():
        embeddings = model.get_embedding(edge_index)

    user_embedding = embeddings[user_id]
    item_embeddings = embeddings[num_users:]  # item embeddings start after users

    scores = torch.matmul(item_embeddings, user_embedding)
    top_scores, top_indices = torch.topk(scores, k=top_k)

    item_ids = item_encoder.inverse_transform(top_indices.cpu().numpy())
    return list(zip(item_ids, top_scores.cpu().numpy()))

# === Main ===
if __name__ == "__main__":
    checkpoint_path = "best_recall_model_1_epoch_110.pth"  # Replace with actual filename
    user_to_query = "B07J3GH1W1"  # Replace with any real user_id from the dataset

    edge_index, user_encoder, item_encoder, num_users, num_items = load_data()
    model = load_model(checkpoint_path, num_nodes=num_users + num_items)

    results = recommend(user_to_query, model, edge_index, user_encoder, item_encoder, num_users, num_items)

    print(f"\nTop Recommendations for user {user_to_query}:")
    for item_id, score in results:
        print(f"{item_id} | score: {score:.4f}")
