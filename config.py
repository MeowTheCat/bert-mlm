from dataclasses import dataclass

@dataclass
class Config:
    MAX_LEN = 20
    BATCH_SIZE = 128
    LR = 0.001
    VOCAB_SIZE = 5834
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1

