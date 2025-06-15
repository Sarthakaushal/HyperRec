from models.lightGCN import get_lightgcn_model
from src.dataloader import load_and_encode_data
from src.loss_metrics import bpr_loss, recall_at_k, ndcg_at_k
import torch
import torch.optim as optim
from src.Enums import *
import os

def train():
    data = load_and_encode_data(
        os.path.join(FilePath.ROOT_DATA_DIR.value, FilePath.TRAIN_DATA.value),
        os.path.join(FilePath.ROOT_DATA_DIR.value, FilePath.VALIDATION_DATA.value),
        os.path.join(FilePath.ROOT_DATA_DIR.value, FilePath.TEST_DATA.value)
        )
    print("Loaded the dataset")
    model = get_lightgcn_model(data['train_data'].num_nodes)
    optimizer = TrainConfig.optim.value(model.parameters(), lr = TrainConfig.lr.value,
                            weight_decay= TrainConfig.decay.value)
    # optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)
    best_recall = 0

    for epoch in range(TrainConfig.epochs.value):
        model.train()
        optimizer.zero_grad()
        embeddings = model.get_embedding(data['train_data'].edge_index)
        loss = bpr_loss(embeddings, 
            data['train_data'].edge_index,data['num_users'], data['num_items'], 
            data['train_df'].groupby('userID')['itemID'].apply(set).to_dict()
        )
        loss.backward()
        optimizer.step()

        if epoch % 2 == 1:
            model.eval()
            with torch.no_grad():
                val_embeddings = model.get_embedding(data['val_data'].edge_index)
                recall = recall_at_k(val_embeddings, data['num_users'], data['num_items'], data['val_df'].groupby('userID')['itemID'].apply(set).to_dict())
                ndcg = ndcg_at_k(val_embeddings, data['num_users'], data['num_items'], data['val_df'].groupby('userID')['itemID'].apply(set).to_dict())
                print(f"Epoch {epoch+1}: Recall@10 = {recall:.4f}, NDCG@10 = {ndcg:.4f}")
