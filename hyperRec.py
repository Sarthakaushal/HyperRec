import torch
import pandas as pd
import numpy as np
import wandb
from torch_geometric.nn import LightGCN
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch.serialization
from src.Enums import FilePath
import torch.nn.functional as F

# === Load LLaVA Embeddings ===
def load_llava_embeddings(path, item_encoder):
    llava_np = np.load(path)
    assert llava_np.shape[0] == len(item_encoder.classes_), "Mismatch in LLaVA embeddings"
    return torch.tensor(llava_np, dtype=torch.float32)

# === Projector ===
class VLLMProjector(torch.nn.Module):
    def __init__(self, input_dim=4096, output_dim=128):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

# === Load LightGCN Only Model ===
def load_lightgcn_model(model_path, num_nodes, pretrained=True):
    model = LightGCN(num_nodes=num_nodes, embedding_dim=128, num_layers=2)
    if pretrained:
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

# === BPR Loss ===
def bpr_loss(embeddings, edge_index, num_users, num_items, user_item_dict):
    user_indices = edge_index[0]
    pos_item_indices = edge_index[1]
    user_embeds = embeddings[user_indices]
    pos_embeds = embeddings[pos_item_indices + num_users]

    neg_indices = []
    for user in user_indices:
        while True:
            neg = np.random.randint(num_items)
            if neg not in user_item_dict.get(user.item(), set()):
                neg_indices.append(neg)
                break
    neg_indices = torch.tensor(neg_indices, device=embeddings.device)
    neg_embeds = embeddings[neg_indices + num_users]

    pos_scores = (user_embeds * pos_embeds).sum(dim=1)
    neg_scores = (user_embeds * neg_embeds).sum(dim=1)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

# === Recall@K ===
def recall_at_k(full_embeddings, edge_index, num_users, num_items, user_item_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = torch.matmul(user_embeds, item_embeds.T)
    _, topk = torch.topk(scores, k=k, dim=1)
    recall = 0
    for u in range(num_users):
        true_items = user_item_dict.get(u, set())
        recs = topk[u].tolist()
        if true_items:
            recall += len(set(recs) & true_items) / len(true_items)
    return recall / num_users

# === NDCG@K ===
def ndcg_at_k(full_embeddings, edge_index, num_users, num_items, user_item_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = torch.matmul(user_embeds, item_embeds.T)
    _, topk = torch.topk(scores, k=k, dim=1)
    ndcg = 0
    for u in range(num_users):
        true_items = user_item_dict.get(u, set())
        recs = topk[u].tolist()
        if true_items:
            dcg = sum((1 / np.log2(i + 2)) for i, it in enumerate(recs) if it in true_items)
            idcg = sum((1 / np.log2(i + 2)) for i in range(min(len(true_items), k)))
            ndcg += (dcg / idcg) if idcg > 0 else 0
    return ndcg / num_users

# === Main ===
if __name__ == "__main__":
    # Configurations
    config = {
        "model_path": "best_recall_model_1_epoch_110.pth",
        "llava_embedding_path": "smolvlm_embeddings.npy",
        "projector_path": "step1_vlm_alignment_smolvlm_dainty-puddle-32.pth",
        "learning_rate": 1e-3,
        "reg_weight": 1e-4,
        "epochs": 30000,
        "patience": 1500
    }
    seed = 120
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check for better alpha value
    train_alpha = False

    model_name = "smolvlm"
    # Initialize W&B
    wandb.init(project="lightgcn_llava_fusion", config=config)
    run = wandb.run.name
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.train.csv.gz")[['user_id', 'parent_asin', 'rating']]
    val_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.valid.csv.gz")[['user_id', 'parent_asin', 'rating']]
    test_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.test.csv.gz")[['user_id', 'parent_asin', 'rating']]
    train_df.columns = val_df.columns = test_df.columns = ['userID', 'itemID', 'rating']
    combined = pd.concat([train_df, val_df, test_df])

    user_enc = LabelEncoder().fit(combined['userID'])
    item_enc = LabelEncoder().fit(combined['itemID'])
    for df in (train_df, val_df):
        df['userID'] = user_enc.transform(df['userID'])
        df['itemID'] = item_enc.transform(df['itemID'])

    num_users = len(user_enc.classes_)
    num_items = len(item_enc.classes_)
    edge_index = torch.tensor([train_df['userID'].values, train_df['itemID'].values], dtype=torch.long).to(device)
    llava_tensor = load_llava_embeddings(cfg.llava_embedding_path, item_enc).to(device)
    train_dict = train_df.groupby('userID')['itemID'].apply(set).to_dict()
    val_dict = val_df.groupby('userID')['itemID'].apply(set).to_dict()

    # Models
    model = load_lightgcn_model(cfg.model_path, num_users + num_items, pretrained=False).to(device)
    projector = VLLMProjector(input_dim=2048 if "smol" in cfg.llava_embedding_path.lower() else 4096).to(device)
    if isinstance(config["projector_path"], str) and config["projector_path"]:
        checkpoint = torch.load(config["projector_path"], map_location=device, weights_only=False)
        projector.load_state_dict(checkpoint["projector_state_dict"])
        print(f"Loaded projector weights from {config['projector_path']}")

    # Optimizer & alpha parameter
    if not isinstance(config["projector_path"], str) and not train_alpha:
        # training the Projector only
        alpha = torch.nn.Parameter(torch.tensor(0.99, device=device))  # existing logic
        optimizer = torch.optim.Adam(list(projector.parameters()),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_weight)
        projector.train()
        for p in model.parameters(): p.requires_grad = False
        alpha.requires_grad = False

    elif train_alpha:  #added: prioritize α-training branch
        # freeze both model & projector
        for p in model.parameters():      p.requires_grad = False
        for p in projector.parameters():  p.requires_grad = False

        # wrap loaded alpha as a learnable Parameter
        loaded_alpha = checkpoint.get("alpha", 0.5)  # fallback
        alpha = torch.nn.Parameter(
            torch.tensor(loaded_alpha, device=device, dtype=torch.float32),
            requires_grad=True
        )  #added

        optimizer = torch.optim.Adam([alpha],
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_weight)

    elif isinstance(config["projector_path"], str):
        # training the Model only
        alpha = checkpoint['alpha']
        optimizer = torch.optim.Adam(list(model.parameters()),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_weight)
        model.train()
        for p in projector.parameters(): p.requires_grad = False

    best_recall = best_ndcg = 0
    epochs_no_imp = 0

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        gcn_embeds = model.get_embedding(edge_index)
        gcn_embeds = F.normalize(gcn_embeds, p=2, dim=1) 
        projected = projector(llava_tensor)
        fused_items = alpha * gcn_embeds[num_users:] + (1 - alpha) * projected
        full_embeds = torch.cat([gcn_embeds[:num_users], fused_items], dim=0)
        loss = bpr_loss(full_embeds, edge_index, num_users, num_items, train_dict)
        reg = gcn_embeds.pow(2).sum() * 1e-4 
        loss+=reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        alpha.data.clamp_(0, 1)  #added: keep α in [0,1]

        if isinstance(alpha, torch.nn.Parameter):
            alpha_val = alpha.item()
        else:
            alpha_val = alpha

        # Log training metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/alpha": alpha_val,
            "train/gcn_norm": gcn_embeds.norm().item(),
            "train/projected_norm": projected.norm().item(),
            "epoch": epoch + 1
        })


        # Validation & early stopping
        if (epoch + 1) % 2 == 0:
            model.eval(); projector.eval()
            with torch.no_grad():
                gcn_embeds = model.get_embedding(edge_index)
                projected = projector(llava_tensor)
                fused_items = alpha * gcn_embeds[num_users:] + (1 - alpha) * projected
                full_embeds = torch.cat([gcn_embeds[:num_users], fused_items], dim=0)
                val_rec = recall_at_k(full_embeds, edge_index, num_users, num_items, val_dict)
                val_ndc = ndcg_at_k(full_embeds, edge_index, num_users, num_items, val_dict)
                val_loss = bpr_loss(full_embeds, edge_index, num_users, num_items, val_dict)
                # Log validation metrics
                wandb.log({
                    "val/loss": val_loss.item(),
                    "val/recall": val_rec,
                    "val/ndcg": val_ndc,
                    "epoch": epoch + 1
                })

                # Early stopping
                improved = False
                if val_rec > best_recall:
                    best_recall = val_rec; improved = True
                if improved and not(isinstance(config["projector_path"], str)) and not train_alpha:
                    print(f"Saved the projector in {epoch}; val/recall= {best_recall}")
                    torch.save({
                        "projector_state_dict": projector.state_dict(),
                        "best_recall": best_recall,
                        "best_ndcg": best_ndcg,
                        "alpha": alpha
                    }, f"step1_vlm_alignment_{model_name}_{run}.pth")
                elif improved and train_alpha:
                    print(f"[{epoch}]Got a improved alpha value {alpha.item()}; val/recall= {best_recall}")
                    # print("Saved the projector")
                    # torch.save({
                    #     "projector_state_dict": projector.state_dict(),
                    #     "best_recall": best_recall,
                    #     "best_ndcg": best_ndcg,
                    #     "alpha": alpha
                    # }, f"step2_model_Training_{model_name}_{run}.pth")
                    
                if val_ndc > best_ndcg:
                    best_ndcg = val_ndc; improved = True
                epochs_no_imp = 0 if improved else epochs_no_imp + 1
                if epochs_no_imp >= cfg.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            if not(isinstance(config["projector_path"], str)) and not train_alpha:
                projector.train()
            elif (isinstance(config["projector_path"], str)) and train_alpha:
                pass
            else:
                model.train()
    # Save the model with step1_vlm alignment
    # if not(isinstance(config["projector_path"], str) and config["projector_path"]):
    #     torch.save({
    #         "projector_state_dict": projector.state_dict(),
    #         "best_recall": best_recall,
    #         "best_ndcg": best_ndcg,
    #         "alpha": alpha
    #     }, f"step1_vlm_alignment_{model_name}_{run}.pth")
    # Finish W&B run
    wandb.finish()
    
