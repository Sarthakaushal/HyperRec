import numpy as np
import torch

def bpr_loss(embeddings, edge_index, num_users, num_items, user_item_dict):
    user_indices = edge_index[0]
    pos_item_indices = edge_index[1]

    user_embeddings = embeddings[user_indices]
    pos_item_embeddings = embeddings[pos_item_indices + num_users]

    neg_item_indices = torch.tensor(
        np.random.choice(num_items, size=user_indices.size(0), replace=True),
        device=embeddings.device
    )
    mask = torch.tensor(
        [neg_item not in user_item_dict.get(user.item(), set()) for user, neg_item in zip(user_indices, neg_item_indices)],
        device=embeddings.device
    )
    neg_item_indices = neg_item_indices[mask]

    neg_item_embeddings = embeddings[neg_item_indices + num_users]
    pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)
    neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos_scores - 2*neg_scores)).mean()
    return loss

def recall_at_k(embeddings, num_users, num_items, user_item_dict, k=10):
    user_indices = torch.arange(num_users, device=embeddings.device)
    item_indices = torch.arange(num_items, device=embeddings.device)
    user_embeddings = embeddings[user_indices]
    item_embeddings = embeddings[item_indices + num_users]
    scores = torch.matmul(user_embeddings, item_embeddings.t())
    _, top_k = torch.topk(scores, k=k, dim=1)

    total_recall = 0
    for user in user_indices:
        true_items = user_item_dict.get(user.item(), set())
        recommended = top_k[user]
        hit = len(set(recommended.cpu().numpy()) & true_items)
        total_recall += hit / len(true_items) if true_items else 0
    return total_recall / num_users

def ndcg_at_k(embeddings, num_users, num_items, user_item_dict, k=10):
    user_indices = torch.arange(num_users, device=embeddings.device)
    item_indices = torch.arange(num_items, device=embeddings.device)
    user_embeddings = embeddings[user_indices]
    item_embeddings = embeddings[item_indices + num_users]
    scores = torch.matmul(user_embeddings, item_embeddings.t())
    _, top_k = torch.topk(scores, k=k, dim=1)

    ndcg = 0
    for user in user_indices:
        true_items = user_item_dict.get(user.item(), set())
        recommended = top_k[user]
        dcg = sum((1 / np.log2(i + 2)) for i, item in enumerate(recommended) if item.item() in true_items)
        idcg = sum((1 / np.log2(i + 2)) for i in range(min(len(true_items), k)))
        ndcg += dcg / idcg if idcg else 0
    return ndcg / num_users