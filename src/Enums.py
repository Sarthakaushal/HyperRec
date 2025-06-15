from enum import Enum
import torch
class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class MetricType(Enum):
    RECALL = "recall"
    NDCG = "ndcg"
    LOSS = "loss"

class ModelType(Enum):
    LIGHT_GCN = "lightgcn"
    LLAVA = "llava"
    FUSION = "fusion"

class HyperParam(Enum):
    EMBEDDING_DIM = "embedding_dim"
    NUM_LAYERS = "num_layers"
    LEARNING_RATE = "learning_rate"
    REG_WEIGHT = "reg_weight"
    EPOCHS = "epochs"

class FilePath(Enum):
    ROOT_DATA_DIR = "/home/spring2024/sk4858/HyperRec/"
    TRAIN_DATA = "data/Books.train.csv.gz"
    VALIDATION_DATA = "data/Books.valid.csv.gz"
    TEST_DATA = "data/Books.test.csv.gz"
    BEST_MODEL = "best_recall_model_1_epoch_110.pth"
    
class TrainConfig(Enum):
    epochs = 300
    lr = 0.01
    decay = 1e-2
    optim = torch.optim.Adam