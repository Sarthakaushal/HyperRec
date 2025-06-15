import torch
import pandas as pd
import numpy as np
import wandb
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import LightGCN
from sklearn.preprocessing import LabelEncoder
from src.Enums import FilePath
import torch.nn.functional as F
import math

def recall_at_k(full_embeddings, num_users, num_items, train_dict, val_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = torch.matmul(user_embeds, item_embeds.T)

    # mask train items
    for u, seen in train_dict.items():
        if seen:
            scores[u, list(seen)] = -1e9

    # get top‑k
    _, topk = torch.topk(scores, k=k, dim=1)

    # only iterate over users with val items
    valid_users = list(val_dict.keys())
    recall = 0.0
    for u in valid_users:
        true_items = val_dict[u]
        recs = set(topk[u].tolist())
        recall += len(recs & true_items) / len(true_items)

    return recall / len(valid_users) 

def ndcg_at_k(full_embeddings, num_users, num_items, train_dict, val_dict, k=10):
    user_embeds = full_embeddings[:num_users]
    item_embeds = full_embeddings[num_users:]
    scores = torch.matmul(user_embeds, item_embeds.T)

    # mask train items
    for u, seen in train_dict.items():
        if seen:
            scores[u, list(seen)] = -1e9

    _, topk = torch.topk(scores, k=k, dim=1)

    valid_users = list(val_dict.keys())
    total_ndcg = 0.0
    for u in valid_users:
        true_items = val_dict[u]
        dcg = 0.0
        for rank, item in enumerate(topk[u].tolist()):
            if item in true_items:
                dcg += 1.0 / math.log2(rank + 2)
        ideal_len = min(len(true_items), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0

    return total_ndcg / len(valid_users)  # <<— divide by #users with val

if __name__ == "__main__":
    # === Config ===
    config = {
        "model_path":    "best_lgcn.pth",  # where we'll save the best model
        "learning_rate": 1e-2,
        "reg_weight":    1e-7,
        "epochs":        1000,
        "patience":      200,
        "eval_every":    2,
        "batch_size":    256,
        "num_neg":       50
    }
    wandb.init(project="lightgcn_mini_batch", config=config)
    cfg = wandb.config

    # === Reproducibility ===
    seed = 120
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Data Loading ===
    train_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.train.csv.gz")[['user_id','parent_asin']]
    val_df   = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.valid.csv.gz")[['user_id','parent_asin']]
    test_df  = pd.read_csv(FilePath.ROOT_DATA_DIR.value + "data/All_Beauty.test.csv.gz")[['user_id','parent_asin']]
    
    train_df.columns = val_df.columns = test_df.columns = ['userID','itemID']
    combined = pd.concat([train_df, val_df, test_df])

    user_enc = LabelEncoder().fit(combined['userID'])
    item_enc = LabelEncoder().fit(combined['itemID'])
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

    # flatten edges for sampling
    all_users = train_df['userID'].values
    all_items = train_df['itemID'].values
    num_edges = len(all_users)

    # === Model, Optimizer & Scheduler ===
    model = LightGCN(num_nodes=num_users+num_items,
                     embedding_dim=64,
                     num_layers=2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.reg_weight
    )
    # warmup for 50 epochs, then constant
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=50
    )

    best_rec, best_ndcg = 0.0, 0.0
    epochs_no_imp = 0

    # === Training ===
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        perm = np.random.permutation(num_edges)
        for start in range(0, num_edges, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            u_batch   = torch.tensor(all_users[idx], device=device)
            pos_batch = torch.tensor(all_items[idx], device=device)

            # sample multi-negatives
            neg_batch = torch.randint(
                0, num_items, (len(idx), cfg.num_neg), device=device
            )
            # avoid sampling positives
            for j, u in enumerate(u_batch):
                mask = neg_batch[j] == pos_batch[j]
                while mask.any():
                    neg_batch[j, mask] = torch.randint(
                        0, num_items, (mask.sum().item(),), device=device
                    )
                    mask = neg_batch[j] == pos_batch[j]

            # edge-dropout for propagation
            drop_ei, _ = dropout_edge(edge_index, p=0.5)
            raw = model.get_embedding(drop_ei)
            gcn = F.normalize(raw, p=2, dim=1)

            # lookup embeddings
            u_emb   = gcn[u_batch]
            pos_emb = gcn[pos_batch + num_users]
            neg_emb = gcn[neg_batch + num_users]  # [B, M, D]

            # compute scores
            pos_scores = (u_emb * pos_emb).sum(dim=1, keepdim=True)    # [B,1]
            neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2)     # [B,M]

            # BPR loss
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

            # batch‐level logging
            wandb.log({
                "train/batch_loss":       loss.item(),
                "train/pos_score_mean":   pos_scores.mean().item(),
                "train/neg_score_mean":   neg_scores.mean().item(),
                "train/pos_score_std":    pos_scores.std().item(),
                "train/neg_score_std":    neg_scores.std().item(),
                "epoch":                  epoch
            })

        # epoch‐level logging
        epoch_loss /= num_edges
        wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

        # LR warmup
        scheduler.step()

        # === Validation ===
        if epoch % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # propagate on train + val edges
                val_ei  = torch.tensor([
                    val_df['userID'].values,
                    val_df['itemID'].values
                ], dtype=torch.long).to(device)
                full_ei = torch.cat([edge_index, val_ei], dim=1)

                raw_full    = model.get_embedding(full_ei)
                full_embeds = F.normalize(raw_full, p=2, dim=1)

                rec  = recall_at_k(full_embeds, num_users, num_items,
                                   train_dict, val_dict, k=10)
                ndcg = ndcg_at_k(full_embeds, num_users, num_items,
                                 train_dict, val_dict, k=10)

            wandb.log({
                "val/recall": rec,
                "val/ndcg":  ndcg,
                "epoch":     epoch
            })

            # save best
            if rec > best_rec:
                best_rec = rec
                best_ndcg = ndcg
                torch.save(model.state_dict(), cfg.model_path)
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1

            if epochs_no_imp >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    wandb.finish()

    # optionally, reload best model:
    model.load_state_dict(torch.load(cfg.model_path))
    print(f"Training complete. Best Recall@10 = {best_rec:.4f}, NDCG@10 = {best_ndcg:.4f}")
    
    # after wandb.finish()
model.load_state_dict(torch.load(cfg.model_path))
with torch.no_grad():
    # full‐graph propagation on train+val edges
    val_ei  = torch.tensor([val_df.userID.values, val_df.itemID.values], device=device)
    full_ei = torch.cat([edge_index, val_ei], dim=1)
    raw_full    = model.get_embedding(full_ei)
    full_embeds = F.normalize(raw_full, p=2, dim=1)

    final_rec  = recall_at_k(full_embeds, num_users, num_items, train_dict, val_dict, k=10)
    final_ndcg = ndcg_at_k(  full_embeds, num_users, num_items, train_dict, val_dict, k=10)

print(f"BEST MODEL → Recall@10: {final_rec:.4f}, NDCG@10: {final_ndcg:.4f}")