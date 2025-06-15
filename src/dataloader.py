import pandas as pd
import torch
import os
import gzip
import json
from src.utils import UnsupportedFileFormatError
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def load_and_encode_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)[['user_id', 'parent_asin', 'rating']]
    val_df = pd.read_csv(val_path)[['user_id', 'parent_asin', 'rating']]
    test_df = pd.read_csv(test_path)[['user_id', 'parent_asin', 'rating']]

    for df in [train_df, val_df, test_df]:
        df.columns = ['userID', 'itemID', 'rating']

    combined_df = pd.concat([train_df, val_df, test_df])
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_encoder.fit(combined_df['userID'])
    item_encoder.fit(combined_df['itemID'])

    for df in [train_df, val_df, test_df]:
        df['userID'] = user_encoder.transform(df['userID'])
        df['itemID'] = item_encoder.transform(df['itemID'])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    def build_data(df):
        edge_index = torch.tensor([df['userID'].values, df['itemID'].values], dtype=torch.long)
        data = Data(edge_index=edge_index)
        data.num_nodes = num_users + num_items
        return data

    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'train_data': build_data(train_df),
        'val_data': build_data(val_df),
        'test_data': build_data(test_df),
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'num_users': num_users,
        'num_items': num_items
    }
    
def toRecBole(dataset_path: str, 
              kv_map: dict, 
              output_dir: str = "dataset", 
              dataset_name: str = "amazon_books",
              split: str = 'train') -> pd.DataFrame:
    """
    Converts an Amazon 5-core JSON dataset to RecBole-compatible format.

    Parameters:
    - dataset_path (str): Path to the .json or .json.gz file.
    - kv_list (dict): Key-Val mapping of the dataset items and the names to use for recbole
    - output_dir (str): Directory to save the formatted dataset.
    - dataset_name (str): Name for the dataset folder (used as dataset name in RecBole).
    - split (str): Split type ('train', 'test', 'valid').
    Output:
    - Creates `output_dir/dataset_name_split/dataset_name_split.inter` file with selected fields.
    
    """
    if dataset_path.endswith('.csv.gz'):
        df = pd.read_csv(dataset_path, compression='gzip',usecols=kv_map.keys())
        print(df.columns)
    else:
        raise ValueError("Unsupported file format. Expected a '.csv.gz' file.")
    # Create output directory structure
    output_path = os.path.join(output_dir, dataset_name+"_"+split)
    os.makedirs(output_path, exist_ok=True)

    # Save to .inter format
    inter_path = os.path.join(output_path, f"{dataset_name}_{split}.inter")
    df.to_csv(inter_path, index=False, sep='\t')

    print(f"Saved RecBole dataset to: {inter_path}")
    return df