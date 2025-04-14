from enum import Enum

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
    TRAIN_DATA = "data/All_Beauty.train.csv.gz"
    VALIDATION_DATA = "data/All_Beauty.valid.csv.gz"
    TEST_DATA = "data/All_Beauty.test.csv.gz"
    BEST_MODEL = "best_recall_model_1_epoch_110.pth"