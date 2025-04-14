import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import LightGCN
import torch.optim as optim
import numpy as np
import wandb
import sys
# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# Initialize wandb
# wandb.init(project="lightGCN-recsys")

# Load the predefined data splits
train_df = pd.read_csv('data/All_Beauty.train.csv.gz')
val_df = pd.read_csv('data/All_Beauty.valid.csv.gz')
test_df = pd.read_csv('data/All_Beauty.test.csv.gz')

# Ensure the columns are named correctly
train_df = train_df[['user_id', 'parent_asin', 'rating']]
train_df.columns = ['userID', 'itemID', 'rating']

val_df = val_df[['user_id', 'parent_asin', 'rating']]
val_df.columns = ['userID', 'itemID', 'rating']

test_df = test_df[['user_id', 'parent_asin', 'rating']]
test_df.columns = ['userID', 'itemID', 'rating']

# Encode user and item IDs as integers
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

# Fit the encoders on the combined data to ensure consistent encoding
combined_df = pd.concat([train_df, val_df, test_df])
user_encoder.fit(combined_df['userID'])
item_encoder.fit(combined_df['itemID'])

train_df['userID'] = user_encoder.transform(train_df['userID'])
train_df['itemID'] = item_encoder.transform(train_df['itemID'])
val_df['userID'] = user_encoder.transform(val_df['userID'])
val_df['itemID'] = item_encoder.transform(val_df['itemID'])
test_df['userID'] = user_encoder.transform(test_df['userID'])
test_df['itemID'] = item_encoder.transform(test_df['itemID'])

# Number of users and items
num_users = combined_df['userID'].nunique()
num_items = combined_df['itemID'].nunique()

# Create user-item dictionaries for training, validation, and test sets
train_user_item_dict = train_df.groupby('userID')['itemID'].apply(set).to_dict()
val_user_item_dict = val_df.groupby('userID')['itemID'].apply(set).to_dict()
test_user_item_dict = test_df.groupby('userID')['itemID'].apply(set).to_dict()
# Create PyG data objects for training, validation, and testing
train_edge_index = torch.tensor([train_df['userID'].values, train_df['itemID'].values], dtype=torch.long)
val_edge_index = torch.tensor([val_df['userID'].values, val_df['itemID'].values], dtype=torch.long)
test_edge_index = torch.tensor([test_df['userID'].values, test_df['itemID'].values], dtype=torch.long)

train_data = Data(edge_index=train_edge_index)
val_data = Data(edge_index=val_edge_index)
test_data = Data(edge_index=test_edge_index)

train_data.num_nodes = num_users + num_items
val_data.num_nodes = num_users + num_items
test_data.num_nodes = num_users + num_items

# Define BPR loss function
def bpr_loss(embeddings, edge_index, num_users, num_items, user_item_dict):
    user_indices = edge_index[0]
    pos_item_indices = edge_index[1]

    user_embeddings = embeddings[user_indices]
    pos_item_embeddings = embeddings[pos_item_indices + num_users]

    # Vectorized negative sampling
    neg_item_indices = torch.tensor(
        np.random.choice(num_items, size=user_indices.size(0), replace=True),
        device=embeddings.device
    )

    # Ensure negative samples are not positive samples
    mask = torch.tensor(
        [neg_item not in user_item_dict.get(user.item(), set()) for user, neg_item in zip(user_indices, neg_item_indices)],
        device=embeddings.device
    )
    neg_item_indices = neg_item_indices[mask]

    neg_item_embeddings = embeddings[neg_item_indices + num_users]

    pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)
    neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    return loss

def recall_at_k(embeddings, edge_index, num_users, num_items, user_item_dict, k=10):
    user_indices = torch.arange(num_users, device=embeddings.device)
    user_embeddings = embeddings[user_indices]

    item_indices = torch.arange(num_items, device=embeddings.device)
    item_embeddings = embeddings[item_indices + num_users]

    scores = torch.matmul(user_embeddings, item_embeddings.t())
    _, top_k_indices = torch.topk(scores, k=k, dim=1)

    recall = 0
    for user in user_indices:
        true_items = user_item_dict.get(user.item(), set())
        recommended_items = top_k_indices[user]
        recall += len(set(recommended_items.cpu().numpy()) & set(true_items)) / len(true_items) if true_items else 0

    return recall / num_users

def ndcg_at_k(embeddings, edge_index, num_users, num_items, user_item_dict, k=10):
    user_indices = torch.arange(num_users, device=embeddings.device)
    user_embeddings = embeddings[user_indices]

    item_indices = torch.arange(num_items, device=embeddings.device)
    item_embeddings = embeddings[item_indices + num_users]

    scores = torch.matmul(user_embeddings, item_embeddings.t())
    _, top_k_indices = torch.topk(scores, k=k, dim=1)

    ndcg = 0
    for user in user_indices:
        true_items = user_item_dict.get(user.item(), set())
        recommended_items = top_k_indices[user]
        dcg = 0
        idcg = 0
        for i, item in enumerate(recommended_items):
            if item.item() in true_items:
                dcg += 1 / np.log2(i + 2)
        for i in range(min(len(true_items), k)):
            idcg += 1 / np.log2(i + 2)
        ndcg += dcg / idcg if idcg > 0 else 0

    return ndcg / num_users

# Define the training function
def train():
    wandb.init()  # Moved init inside train to capture sweep config
    config = wandb.config
    # Initialize LightGCN model using config values
    model = LightGCN(num_nodes=train_data.num_nodes, embedding_dim=config.embedding_dim, num_layers=config.num_layers)
    wandb.watch(model, log="all", log_freq=10)  # Watch the model for logging gradients/weights

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.reg_weight)

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        embeddings = model.get_embedding(train_data.edge_index)
        loss = bpr_loss(embeddings, train_data.edge_index, num_users, num_items, train_user_item_dict)
        loss.backward()
        optimizer.step()

        # Log training loss to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": loss.item()})

        if epoch % 2 == 1:  # Validate every 2 epochs
            val_embeddings = model.get_embedding(val_data.edge_index)
            val_recall = recall_at_k(val_embeddings, val_data.edge_index, num_users, num_items, val_user_item_dict)
            val_ndcg = ndcg_at_k(val_embeddings, val_data.edge_index, num_users, num_items, val_user_item_dict)
            val_loss = bpr_loss(val_embeddings, val_data.edge_index, num_users, num_items, val_user_item_dict)

            # Log validation metrics to wandb
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss.item(), "val_recall@10": val_recall, "val_ndcg@10": val_ndcg})

            print(f"Epoch {epoch+1}/{config.epochs}, Validation Loss: {val_loss.item()}, Validation Recall@10: {val_recall:.4f}, Validation NDCG@10: {val_ndcg:.4f}")

        # if epoch % 10 == 3:  # Test every 10 epochs
            test_embeddings = model.get_embedding(test_data.edge_index)
            test_recall = recall_at_k(test_embeddings, test_data.edge_index, num_users, num_items, test_user_item_dict)
            test_ndcg = ndcg_at_k(test_embeddings, test_data.edge_index, num_users, num_items, test_user_item_dict)

            # Log test metrics to wandb
            wandb.log({"epoch": epoch + 1, "test_recall@10": test_recall, "test_ndcg@10": test_ndcg})

            print(f"Epoch {epoch+1}/{config.epochs}, Test Recall@10: {test_recall:.4f}, Test NDCG@10: {test_ndcg:.4f}")

        recall = recall_at_k(embeddings, train_data.edge_index, num_users, num_items, train_user_item_dict)
        ndcg = ndcg_at_k(embeddings, train_data.edge_index, num_users, num_items, train_user_item_dict)

        # Log training metrics to wandb
        wandb.log({"epoch": epoch + 1, "train_recall@10": recall, "train_ndcg@10": ndcg})

        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {loss.item()}, Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',  # Replace with the actual metric you're optimizing
        'goal': 'minimize'
    },
    'parameters': {
        'embedding_dim': {
            'values': [32, 64, 128, 256]
        },
        'num_layers': {
            'values': [1,2,3,4]  # Example values to sweep over
        },
        'reg_weight': {
            'values': [1e-4, 1e-3, 1e-2]
        },
        'learning_rate': {
            'values': [0.001, 0.01, 0.1]
        },
        'epochs': {
            'value': 150  # Fixed value
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="lightGCN-recsys")

# Run the sweep
wandb.agent(sweep_id, train)
