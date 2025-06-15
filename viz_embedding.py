import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import math
from hyperRec_v2 import VLLMProjector, load_lightgcn_model, recall_at_k, ndcg_at_k
from src.Enums import FilePath
from sklearn.metrics.pairwise import cosine_similarity

# Function to get embeddings for all users and items
def get_all_embeddings(model, projector, llava_tensor, edge_index, num_users, alpha):
    model.eval()
    projector.eval()

    raw_full = model.get_embedding(edge_index)
    gcn_full = F.normalize(raw_full, p=2, dim=1)
    proj_full = projector(llava_tensor)

    fused_full = alpha * gcn_full[num_users:] + (1 - alpha) * proj_full
    full_embeds = torch.cat([gcn_full[:num_users], fused_full], dim=0)
    return gcn_full[:num_users], gcn_full[num_users:], fused_full

if __name__ == "__main__":
    config = {
        "model_path":   "llava_best_fusion.pth",  # Path to the saved model checkpoint
        "llava_np":     "llava_embeddings_ordered.npy",  # Path to the LLaVA embeddings
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    ckpt = torch.load(config['model_path'], map_location=device)

    # Load the LLaVA embeddings
    llava_np = np.load(config['llava_np'])
    embedding_dim = llava_np.shape[1]
    llava_tensor = torch.tensor(llava_np, dtype=torch.float32, device=device)

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
    num_items = len(item_enc.classes_)
    edge_index = torch.tensor([
        train_df['userID'].values,
        train_df['itemID'].values + num_users  # Adjust item indices for the graph
    ], dtype=torch.long).to(device)

    # Initialize the model and projector
    model = load_lightgcn_model(config['model_path'], num_users + num_items, pretrained=False).to(device)
    projector = VLLMProjector(input_dim=embedding_dim, output_dim=64).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    projector.load_state_dict(ckpt['projector_state_dict'])
    alpha_param = ckpt['alpha'].to(device)
    alpha = alpha_param.item() # Get the scalar value of alpha

    # Get the embeddings
    gcn_user_embeddings, gcn_item_embeddings, fused_item_embeddings = get_all_embeddings(
        model, projector, llava_tensor, edge_index, num_users, alpha
    )

    # --- Define pairs of similar and different items (YOU NEED TO CHOOSE THESE) ---
    # You should replace these placeholders with actual item IDs from your dataset
    # and ensure they are present in your 'all_items' LabelEncoder.classes_.

    similar_item_ids = ["B08XJWLLKQ", "B08QKXY1KN"]  # two conditioners
    different_item_ids = ["B08XJWLLKQ", "B0BC273JJR"] # ["Tattoo Eyebrow Stickers, Waterproof Eyebrow, 4D Imitation Eyebrow Tattoos, 4D Hair-like Authentic Eyebrows Waterproof Long Lasting for Woman & Man Makeup Tool", "Precision Plunger Bars for Cartridge Grips \u2013 93mm \u2013 Bag of 10 Plungers"]

    # Function to get the embedding vector for an item ID
    def get_item_embedding_by_id(item_id, item_enc, item_embeddings_tensor):
        try:
            encoded_id = np.where(item_enc.classes_ == item_id)[0][0]
            return item_embeddings_tensor[encoded_id].cpu().detach().numpy().reshape(1, -1)
        except IndexError:
            print(f"Warning: Item ID '{item_id}' not found in the encoder.")
            return None

    # Get embeddings for the chosen items (using fused embeddings)
    embed_item1_similar = get_item_embedding_by_id(similar_item_ids[0], item_enc, fused_item_embeddings)
    embed_item2_similar = get_item_embedding_by_id(similar_item_ids[1], item_enc, fused_item_embeddings)
    embed_item1_different = get_item_embedding_by_id(different_item_ids[0], item_enc, fused_item_embeddings)
    embed_item2_different = get_item_embedding_by_id(different_item_ids[1], item_enc, fused_item_embeddings)

    if embed_item1_similar is not None and embed_item2_similar is not None:
        similarity_similar = cosine_similarity(embed_item1_similar, embed_item2_similar)[0][0]
        print(f"Cosine Similarity between '{similar_item_ids[0]}' and '{similar_item_ids[1]}': {similarity_similar:.4f}")
    else:
        print("Could not calculate similarity for similar items due to missing embeddings.")

    if embed_item1_different is not None and embed_item2_different is not None:
        similarity_different = cosine_similarity(embed_item1_different, embed_item2_different)[0][0]
        print(f"Cosine Similarity between '{different_item_ids[0]}' and '{different_item_ids[1]}': {similarity_different:.4f}")
    else:
        print("Could not calculate similarity for different items due to missing embeddings.")