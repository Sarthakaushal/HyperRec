import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import math
from hyperRec_v2 import VLLMProjector, load_lightgcn_model, recall_at_k, ndcg_at_k
from src.Enums import FilePath

# Function to perform inference from the checkpoint
def infer_recommendations(model, projector, llava_tensor, edge_index, num_users, num_items, alpha, user_enc, item_enc, top_k=10):
    model.eval()
    projector.eval()
    
    raw_full = model.get_embedding(edge_index)
    gcn_full = F.normalize(raw_full, p=2, dim=1)
    proj_full = projector(llava_tensor)
    
    fused_full = alpha * gcn_full[num_users:] + (1 - alpha) * proj_full
    full_embeds = torch.cat([gcn_full[:num_users], fused_full], dim=0)

    user_id = 0  # Change to the user ID you want to perform inference for
    user_embedding = full_embeds[user_id]
    
    scores = torch.matmul(user_embedding, full_embeds[num_users:].T)
    _, top_k_items = torch.topk(scores, k=top_k)
    
    top_k_item_ids = item_enc.inverse_transform(top_k_items.cpu().numpy())
    
    return top_k_item_ids

# Function to evaluate the model
def evaluate_model(model, projector, llava_tensor, edge_index, num_users, num_items, alpha, train_dict, val_dict, top_k=10):
    model.eval()
    projector.eval()

    raw_full = model.get_embedding(edge_index)
    gcn_full = F.normalize(raw_full, p=2, dim=1)
    proj_full = projector(llava_tensor)
    
    fused_full = alpha * gcn_full[num_users:] + (1 - alpha) * proj_full
    full_embeds = torch.cat([gcn_full[:num_users], fused_full], dim=0)

    # Calculate recall@k and NDCG@k on the validation set
    recall = recall_at_k(full_embeds, num_users, num_items, train_dict, val_dict, k=top_k)
    ndcg = ndcg_at_k(full_embeds, num_users, num_items, train_dict, val_dict, k=top_k)

    return recall, ndcg

# Main script for training and inference
if __name__ == "__main__":
    config = {
        "model_path":   "llava_best_fusion.pth",  # Path to the saved model checkpoint
        "llava_np":     "embeddings/llava_embeddings_ordered_4b.npy",  # Path to the LLaVA embeddings
    }
    print("*"*100,'\n',config['llava_np'], '\n',"*"*100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    ckpt = torch.load(config['model_path'], map_location=device)
    
    # Print out the keys of the model state dict to ensure we're accessing them correctly
    # print(ckpt['model_state_dict'].keys())  # This should show 'embedding.weight'

    # Extract embedding weights and split into user and item embeddings
    embedding_weight = ckpt['model_state_dict']['embedding.weight']
    
    # Get the number of users and items by splitting the embedding weights
    num_users = len(embedding_weight) // 2  # Assuming equal split
    num_items = num_users  # Adjust accordingly if num_users != num_items
    
    # Split the embeddings into user and item embeddings
    user_embeds = embedding_weight[:num_users]
    item_embeds = embedding_weight[num_users:]

    print(f"User Embeddings Shape: {user_embeds.shape}")
    print(f"Item Embeddings Shape: {item_embeds.shape}")

    # Load the LLaVA embeddings
    llava_np = np.load(config['llava_np'])
    embedding_dim = llava_np.shape[1]
    llava_tensor = torch.tensor(llava_np, dtype=torch.float32, device=device)
    
    # Load the training, validation, and test data
    train_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value+"data/All_Beauty.train.csv.gz")[['user_id', 'parent_asin']]
    val_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value+"data/All_Beauty.valid.csv.gz")[['user_id', 'parent_asin']]
    test_df = pd.read_csv(FilePath.ROOT_DATA_DIR.value+"data/All_Beauty.test.csv.gz")[['user_id', 'parent_asin']]

    # Rename columns uniformly
    for df in (train_df, val_df, test_df):
        df.columns = ['userID', 'itemID']

    # Fit a single encoder on ALL users and ALL items across train, val, and test
    all_users = pd.concat([train_df['userID'], val_df['userID'], test_df['userID']])
    all_items = pd.concat([train_df['itemID'], val_df['itemID'], test_df['itemID']])

    # Fit LabelEncoders for users and items
    user_enc = LabelEncoder().fit(all_users)
    item_enc = LabelEncoder().fit(all_items)

    # Transform train/val splits
    for df in (train_df, val_df):
        df['userID'] = user_enc.transform(df['userID'])
        df['itemID'] = item_enc.transform(df['itemID'])

    # Create edge_index for training data
    edge_index = torch.tensor([
        train_df['userID'].values,
        train_df['itemID'].values
    ], dtype=torch.long).to(device)
    
    # Create dictionaries for training and validation data
    train_dict = train_df.groupby('userID')['itemID'].apply(set).to_dict()
    val_dict = val_df.groupby('userID')['itemID'].apply(set).to_dict()
    
    # Initialize the model
    model = load_lightgcn_model(config['model_path'], num_users + num_items, pretrained=False).to(device)
    projector = VLLMProjector(input_dim=embedding_dim, output_dim=64).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    projector.load_state_dict(ckpt['projector_state_dict'])
    alpha = torch.nn.Parameter(ckpt['alpha'].to(device))
    
    # Evaluate the model on the validation set
    recall, ndcg = evaluate_model(model, projector, llava_tensor, edge_index, num_users, num_items, alpha, train_dict, val_dict, top_k=10)
    
    print(f"Validation Recall@10: {recall:.4f}")
    print(f"Validation NDCG@10: {ndcg:.4f}")
