from src.dataloader import toRecBole

# Step1 :all I need is to create a mapping of the columns in the dataset to the names used in recbole
cols_to_use = {
    'user_id': 'user_id',
    'parent_asin': 'item_id',
    # 'rating': 'rating',
    'timestamp': 'timestamp'
}

# Step2 : Call the recbole conversion function

toRecBole(dataset_path='data/Books.train.csv.gz', kv_map= cols_to_use,
          dataset_name='amazon_book', split='train')
toRecBole(dataset_path='data/Books.test.csv.gz', kv_map= cols_to_use,
          dataset_name='amazon_book', split='test')
toRecBole(dataset_path='data/Books.train.csv.gz', kv_map= cols_to_use,
          dataset_name='amazon_book', split='valid')