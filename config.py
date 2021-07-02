from dataclasses import dataclass

@dataclass
class Config:
    MAX_LEN = 22
    BATCH_SIZE = 128
    LR = 0.1
    VOCAB_SIZE = 5836
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1
