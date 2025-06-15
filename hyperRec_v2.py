import torch
import numpy as np
import pandas as pd
import wandb
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import LightGCN
import torch.nn.functional as F
import math
from sklearn.preprocessing import LabelEncoder
from src.Enums import FilePath

# === Fusion components from hyperRec.py hyperRec.py.txt](file-service://file-APvLZdKyXG7Cth7y8T4yth) ===
class VLLMProjector(torch.nn.Module):
    def __init__(self, input_dim=2048, output_dim=64):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim)
        )
    def forward(self, x):
        return self.projection(x)

def load_lightgcn_model(model_path, num_nodes, pretrained=True):
    model = LightGCN(num_nodes=num_nodes, embedding_dim=64, num_layers=2)
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model

# === Masked metrics from final_training lightGCN_final_training.py.txt](file-service://file-FDrLNP7T5wKNxDsrjBrvc3) ===
def recall_at_k(full_embeddings, num_users, num_items, train_dict, val_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = user_embeds @ item_embeds.T
    for u, seen in train_dict.items():
        if seen: scores[u, list(seen)] = -1e9
    _, topk = torch.topk(scores, k=k, dim=1)
    valid = list(val_dict.keys())
    total = 0.0
    for u in valid:
        trues = val_dict[u]
        recs  = set(topk[u].tolist())
        total += len(recs & trues) / len(trues)
    return total / len(valid)

def ndcg_at_k(full_embeddings, num_users, num_items, train_dict, val_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = user_embeds @ item_embeds.T
    for u, seen in train_dict.items():
        if seen: scores[u, list(seen)] = -1e9
    _, topk = torch.topk(scores, k=k, dim=1)
    valid = list(val_dict.keys())
    total = 0.0
    for u in valid:
        trues = val_dict[u]
        dcg = 0.0
        for r, it in enumerate(topk[u].tolist()):
            if it in trues:
                dcg += 1.0 / math.log2(r+2)
        ideal = min(len(trues), k)
        idcg  = sum(1.0 / math.log2(i+2) for i in range(ideal))
        total += (dcg/idcg) if idcg>0 else 0.0
    return total / len(valid)

def _get_model_id( hf_id):
        if "llava" in hf_id.lower():
            return 'llava'
        elif "smolvlm" in hf_id.lower():
            return 'smolvlm'
        elif "qwen" in hf_id.lower():
            return "qwen"
        elif "clip" in hf_id.lower():
            return "clip"
        else:
            raise ValueError(f"Unsupported model_id: {hf_id}")
        
# === Main fusion training ===
if __name__ == "__main__":
    # -- Config
    config = {
        "model_path":   "best_fusion.pth",
        "light_gcn_best": "best_lgcn.pth",
        "llava_np":     "embeddings/qwen_embeddings_ordered_8b.npy",
        "learning_rate":1e-3,
        "reg_weight":   1e-6,
        "epochs":       1000,
        "patience":     200,
        "eval_every":   2,
        "batch_size":   256,
        "num_neg":      20
    }
    print(f"Config of the embeddings are {config['llava_np']}")
    torch.manual_seed(120)
    torch.cuda.manual_seed_all(120)
    
    model_id = _get_model_id(config['llava_np'])
    config["model_save_pth"] = model_id+"_"+config['model_path']
    wandb.init(project="hyperRec_fusion", config=config)
    cfg = wandb.config
    
    # -- Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load LLaVA embeddings (already extracted)
    llava_np = np.load(cfg.llava_np)
    llava_tensor = torch.tensor(llava_np, dtype=torch.float32, device=device)
    embedding_dim = llava_np.shape[1]

    # -- Data loading & unified encoding
    train_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.train.csv.gz")[['user_id','parent_asin']]
    val_df   = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.valid.csv.gz")[['user_id','parent_asin']]
    test_df  = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.test.csv.gz")[['user_id','parent_asin']]

    # rename columns uniformly
    for df in (train_df, val_df, test_df):
        df.columns = ['userID','itemID']

    # fit a single encoder on ALL items and ALL users
    all_users = pd.concat([train_df['userID'], val_df['userID'], test_df['userID']])
    all_items = pd.concat([train_df['itemID'], val_df['itemID'], test_df['itemID']])
    user_enc = LabelEncoder().fit(all_users)
    item_enc = LabelEncoder().fit(all_items)

    # transform train/val splits
    for df in (train_df, val_df):
        df['userID'] = user_enc.transform(df['userID'])
        df['itemID'] = item_enc.transform(df['itemID'])

    num_users = len(user_enc.classes_)   # now includes any users in test set too
    num_items = len(item_enc.classes_)   # matches llava_np.shape[0]
    edge_index = torch.tensor([
        train_df['userID'].values,
        train_df['itemID'].values
    ], dtype=torch.long).to(device)

    train_dict = train_df.groupby('userID')['itemID'].apply(set).to_dict()
    val_dict   = val_df.groupby('userID')['itemID'].apply(set).to_dict()

    all_users = train_df['userID'].values
    all_items = train_df['itemID'].values
    num_edges = len(all_users)

    # -- Models, projector & α
    model     = load_lightgcn_model(cfg.light_gcn_best, num_users+num_items, pretrained=True).to(device)
    projector = VLLMProjector(input_dim=embedding_dim, output_dim=64).to(device)
    # load pretrained projector if you have one:
    # ckpt = torch.load("step1_vlm_alignment_....pth", map_location=device)
    # projector.load_state_dict(ckpt["projector_state_dict"])

    alpha = torch.nn.Parameter(torch.tensor(0.5, device=device), requires_grad=True)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()) + [alpha],
        lr=cfg.learning_rate,
        weight_decay=cfg.reg_weight
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=50)

    best_rec = best_ndcg = 0.0
    epochs_no_imp = 0

    # -- Training loop
    for epoch in range(1, cfg.epochs+1):
        model.train(); projector.train()
        epoch_loss = 0.0

        perm = np.random.permutation(num_edges)
        for start in range(0, num_edges, cfg.batch_size):
            idx = perm[start:start+cfg.batch_size]
            u_b = torch.tensor(all_users[idx], device=device)
            p_b = torch.tensor(all_items[idx], device=device)

            # multi-negative sampling
            n_b = torch.randint(0, num_items, (len(idx), cfg.num_neg), device=device)
            for j,u in enumerate(u_b):
                mask = n_b[j]==p_b[j]
                while mask.any():
                    n_b[j,mask] = torch.randint(0, num_items, (mask.sum().item(),), device=device)
                    mask = n_b[j]==p_b[j]

            # edge-dropout + GCN embed + normalize
            drop_ei,_ = dropout_edge(edge_index, p=0.5)
            raw = model.get_embedding(drop_ei)
            gcn = F.normalize(raw, p=2, dim=1)

            # fusion
            proj = projector(llava_tensor)
            gi   = gcn[num_users:]
            fused= alpha*gi + (1-alpha)*proj
            full = torch.cat([gcn[:num_users], fused], dim=0)

            # lookup
            u_e = full[u_b]
            p_e = full[p_b + num_users]
            n_e = full[n_b + num_users]

            pos_scores = (u_e * p_e).sum(dim=1, keepdim=True)
            neg_scores = (u_e.unsqueeze(1) * n_e).sum(dim=2)

            # BPR loss + reg
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            loss = loss + raw.pow(2).sum() * cfg.reg_weight

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/alpha":      alpha.item(),
                "epoch":            epoch
            })

        wandb.log({"train/epoch_loss": epoch_loss/num_edges, "epoch": epoch})
        scheduler.step()

        # -- Validation
        if epoch % cfg.eval_every == 0:
            model.eval(); projector.eval()
            with torch.no_grad():
                val_ei = torch.tensor([
                    val_df['userID'].values,
                    val_df['itemID'].values
                ], dtype=torch.long).to(device)
                full_ei = torch.cat([edge_index, val_ei], dim=1)

                raw_full    = model.get_embedding(full_ei)
                gcn_full    = F.normalize(raw_full, p=2, dim=1)
                proj_full   = projector(llava_tensor)
                fused_full  = alpha*gcn_full[num_users:] + (1-alpha)*proj_full
                full_embeds = torch.cat([gcn_full[:num_users], fused_full], dim=0)

                rec  = recall_at_k(full_embeds, num_users, num_items, train_dict, val_dict)
                ndcg = ndcg_at_k(  full_embeds, num_users, num_items, train_dict, val_dict)

            wandb.log({"val/recall": rec, "val/ndcg": ndcg, "epoch": epoch})

            if rec > best_rec:
                best_rec, best_ndcg = rec, ndcg
                print(f"Saved best model with rec:{rec} and ndcg: {ndcg}")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "projector_state_dict": projector.state_dict(),
                    "alpha": alpha.detach().cpu(),
                    "best_recall": rec,
                    "best_ndcg": ndcg
                }, cfg.model_save_pth)
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1

            if epochs_no_imp >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    wandb.finish()

    # -- Final evaluation of best checkpoint
    ckpt = torch.load(cfg.model_save_pth, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    projector.load_state_dict(ckpt["projector_state_dict"])
    alpha.data = ckpt["alpha"].to(device)
    model.eval(); projector.eval()
    with torch.no_grad():
        val_ei = torch.tensor([val_df['userID'].values, val_df['itemID'].values], device=device)
        full_ei = torch.cat([edge_index, val_ei], dim=1)
        raw_full    = model.get_embedding(full_ei)
        gcn_full    = F.normalize(raw_full, p=2, dim=1)
        proj_full   = projector(llava_tensor)
        fused_full  = alpha*gcn_full[num_users:] + (1-alpha)*proj_full
        full_embeds = torch.cat([gcn_full[:num_users], fused_full], dim=0)

        final_rec  = recall_at_k(full_embeds, num_users, num_items, train_dict, val_dict)
        final_ndcg = ndcg_at_k(  full_embeds, num_users, num_items, train_dict, val_dict)

    print(f"BEST MODEL → Recall@10: {final_rec:.4f}, NDCG@10: {final_ndcg:.4f}")