import wandb
from models.lightgcn_model import get_lightgcn_model
from src.dataloader import load_and_encode_data
from src.loss_metrics import bpr_loss, recall_at_k, ndcg_at_k
import torch

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'embedding_dim': {'values': [32, 64, 128]},
        'num_layers': {'values': [1, 2, 3]},
        'learning_rate': {'values': [0.001, 0.01]},
        'reg_weight': {'values': [1e-4, 1e-3]},
        'epochs': {'value': 150}
    }
}

def train():
    wandb.init()
    config = wandb.config
    data = load_and_encode_data("data/All_Beauty.train.csv.gz", "data/All_Beauty.valid.csv.gz", "data/All_Beauty.test.csv.gz")
    model = get_lightgcn_model(data['train_data'].num_nodes, config.embedding_dim, config.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.reg_weight)

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.get_embedding(data['train_data'].edge_index)
        loss = bpr_loss(embeddings, data['train_data'].edge_index, data['num_users'], data['num_items'], data['train_df'].groupby('userID')['itemID'].apply(set).to_dict())
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                val_emb = model.get_embedding(data['val_data'].edge_index)
                recall = recall_at_k(val_emb, data['num_users'], data['num_items'], data['val_df'].groupby('userID')['itemID'].apply(set).to_dict())
                ndcg = ndcg_at_k(val_emb, data['num_users'], data['num_items'], data['val_df'].groupby('userID')['itemID'].apply(set).to_dict())
                wandb.log({"epoch": epoch, "val_loss": loss.item(), "recall@10": recall, "ndcg@10": ndcg})

sweep_id = wandb.sweep(sweep_config, project="lightGCN-recsys")
wandb.agent(sweep_id, train)