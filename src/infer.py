from models.lightgcn_model import get_lightgcn_model
from src.dataloader import load_and_encode_data
import torch

if __name__ == "__main__":
    data = load_and_encode_data("data/All_Beauty.train.csv.gz", "data/All_Beauty.valid.csv.gz", "data/All_Beauty.test.csv.gz")
    model = get_lightgcn_model(data['train_data'].num_nodes)
    model.load_state_dict(torch.load("best_recall_model_1_epoch_110.pth", map_location="cpu"))
    model.eval()

    user_raw_id = "B07J3GH1W1"
    if user_raw_id not in data['user_encoder'].classes_:
        print("User not found")
    else:
        user_id = data['user_encoder'].transform([user_raw_id])[0]
        with torch.no_grad():
            embeddings = model.get_embedding(data['train_data'].edge_index)
            scores = torch.matmul(embeddings[user_id], embeddings[data['num_users']:] .T)
            top_k = torch.topk(scores, k=10).indices
            recs = data['item_encoder'].inverse_transform(top_k.cpu().numpy())
            print(f"Recommendations for {user_raw_id}:", recs)