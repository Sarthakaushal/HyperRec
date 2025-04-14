import pandas as pd
import torch
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