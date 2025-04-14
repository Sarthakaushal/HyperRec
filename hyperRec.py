# train_lightgcn_llava_fused.py

import torch
import pandas as pd
import numpy as np
from torch_geometric.nn import LightGCN
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch.serialization

# === Load LLaVA Embeddings ===
def load_llava_embeddings(path, item_encoder):
    llava_np = np.load(path)
    assert llava_np.shape[0] == len(item_encoder.classes_), "Mismatch in LLaVA embeddings"
    return torch.tensor(llava_np, dtype=torch.float32)

# === Projector ===
class LlavaProjector(torch.nn.Module):
    def __init__(self, input_dim=4096, output_dim=128):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

# === Fusion Layer ===
class FusionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# === Load LightGCN Only Model ===
def load_lightgcn_model(model_path, num_nodes):
    model = LightGCN(num_nodes=num_nodes, embedding_dim=128, num_layers=2)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# === BPR Loss with filtered negatives ===
def bpr_loss(embeddings, edge_index, num_users, num_items, user_item_dict):
    user_indices = edge_index[0]
    pos_item_indices = edge_index[1]

    user_embeddings = embeddings[user_indices]
    pos_item_embeddings = embeddings[pos_item_indices + num_users]

    neg_item_indices = []
    for user in user_indices:
        while True:
            neg = np.random.randint(num_items)
            if neg not in user_item_dict.get(user.item(), set()):
                neg_item_indices.append(neg)
                break

    neg_item_indices = torch.tensor(neg_item_indices, device=embeddings.device)
    neg_item_embeddings = embeddings[neg_item_indices + num_users]

    pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)
    neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)

    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean() + abs(neg_scores.mean())

# === Recall@K ===
def recall_at_k(full_embeddings, edge_index, num_users, num_items, user_item_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = torch.matmul(user_embeds, item_embeds.T)
    _, topk = torch.topk(scores, k=k, dim=1)

    recall = 0
    for user in range(num_users):
        true_items = user_item_dict.get(user, set())
        rec_items = topk[user].tolist()
        if true_items:
            recall += len(set(rec_items) & true_items) / len(true_items)

    return recall / num_users

# === NDCG@K ===
def ndcg_at_k(full_embeddings, edge_index, num_users, num_items, user_item_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = torch.matmul(user_embeds, item_embeds.T)
    _, topk = torch.topk(scores, k=k, dim=1)

    ndcg = 0
    for user in range(num_users):
        true_items = user_item_dict.get(user, set())
        rec_items = topk[user].tolist()
        if true_items:
            dcg, idcg = 0, 0
            for i, item in enumerate(rec_items):
                if item in true_items:
                    dcg += 1 / np.log2(i + 2)
            for i in range(min(len(true_items), k)):
                idcg += 1 / np.log2(i + 2)
            ndcg += dcg / idcg if idcg else 0

    return ndcg / num_users

# === Main Entry ===
if __name__ == "__main__":
    model_path = "best_recall_model_1_epoch_110.pth"
    projector_path = "best_fused_model_frozen_gcn.pth"
    llava_embedding_path = "llava_item_embeddings.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess
    train_df = pd.read_csv("data/All_Beauty.train.csv.gz")[['user_id', 'parent_asin', 'rating']]
    val_df = pd.read_csv("data/All_Beauty.valid.csv.gz")[['user_id', 'parent_asin', 'rating']]
    test_df = pd.read_csv("data/All_Beauty.test.csv.gz")[['user_id', 'parent_asin', 'rating']]
    
    train_df.columns = val_df.columns = test_df.columns = ['userID', 'itemID', 'rating']
    combined_df = pd.concat([train_df, val_df, test_df])

    user_encoder = LabelEncoder().fit(combined_df['userID'])
    item_encoder = LabelEncoder().fit(combined_df['itemID'])

    train_df['userID'] = user_encoder.transform(train_df['userID'])
    train_df['itemID'] = item_encoder.transform(train_df['itemID'])
    val_df['userID'] = user_encoder.transform(val_df['userID'])
    val_df['itemID'] = item_encoder.transform(val_df['itemID'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    edge_index = torch.tensor([train_df['userID'].values, train_df['itemID'].values], dtype=torch.long).to(device)
    llava_tensor = load_llava_embeddings(llava_embedding_path, item_encoder).to(device)
    train_user_item_dict = train_df.groupby('userID')['itemID'].apply(set).to_dict()
    val_user_item_dict = val_df.groupby('userID')['itemID'].apply(set).to_dict()

    model = load_lightgcn_model(model_path, num_nodes=num_users + num_items).to(device)

    projector = LlavaProjector().to(device)
    fusion_layer = FusionLayer(input_dim=256, output_dim=128).to(device)

    with torch.serialization.safe_globals({"sklearn.preprocessing._label": {"LabelEncoder": LabelEncoder}}):
        projector_ckpt = torch.load(projector_path, map_location="cpu", weights_only=False)
    projector.load_state_dict(projector_ckpt['projector_state_dict'])

    learning_rate = 1e-2  # Reduced learning rate
    reg_weight = 1e-5  # Reduced weight decay
    epochs = 3000

    # Early stopping parameters
    patience = 50  # Number of epochs to wait for improvement
    best_recall = 0.0
    best_ndcg = 0.0
    epochs_without_improvement = 0

    # Set both LightGCN and projector to training mode
    model.train()
    projector.train()
    # Define a learnable parameter alpha
    alpha = torch.nn.Parameter(torch.tensor(0.999, device=device))  # Initialize alpha to 0.5
    # Adjust optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()) + [alpha],
        lr=learning_rate,
        weight_decay=reg_weight
    )

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Get embeddings
        gcn_embeds = model.get_embedding(edge_index)
        projected_llava = projector(llava_tensor)

        # Debug embeddings
        print(f"Epoch {epoch+1} | GCN Embeddings Norm: {gcn_embeds.norm():.4f}")
        print(f"Epoch {epoch+1} | Projected LLaVA Norm: {projected_llava.norm():.4f}")

        

        # Merge the embeddings using alpha
        fused_item_embeds = alpha * gcn_embeds[num_users:] + (1 - alpha) * projected_llava
        # Fuse embeddings
        # fused_item_embeds = fusion_layer(torch.cat((gcn_embeds[num_users:], projected_llava), dim=-1))
        full_embeddings = torch.cat([gcn_embeds[:num_users], fused_item_embeds], dim=0)

        # Compute loss
        loss = bpr_loss(full_embeddings, edge_index, num_users, num_items, train_user_item_dict)

        # Debug loss
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # Backpropagation
        loss.backward()

        # Debug gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm():.4f}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Validation
        if epoch % 2 == 1:
            model.eval()
            projector.eval()
            with torch.no_grad():
                gcn_embeds = model.get_embedding(edge_index)
                projected_llava = projector(llava_tensor)
                fused_item_embeds = fusion_layer(torch.cat((gcn_embeds[num_users:], projected_llava), dim=-1))
                full_embeds = torch.cat([gcn_embeds[:num_users], fused_item_embeds], dim=0)

                val_recall = recall_at_k(full_embeds, edge_index, num_users, num_items, val_user_item_dict)
                val_ndcg = ndcg_at_k(full_embeds, edge_index, num_users, num_items, val_user_item_dict)

                print(f"Epoch {epoch+1:02d} | Validation Recall@10: {val_recall:.4f} | Validation NDCG@10: {val_ndcg:.4f}")

                # Check for improvement
                if val_recall > best_recall or val_ndcg > best_ndcg:
                    if val_recall > best_recall:
                        print(f"Epoch {epoch+1:02d} | New Best Recall@10: {val_recall:.4f}")
                        best_recall = val_recall
                    if val_ndcg > best_ndcg:
                        print(f"Epoch {epoch+1:02d} | New Best NDCG@10: {val_ndcg:.4f}")
                        best_ndcg = val_ndcg
                    epochs_without_improvement = 0  # Reset patience counter
                else:
                    epochs_without_improvement += 1

                # Early stopping condition
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1:02d}.")
                    break

            model.train()
            projector.train()