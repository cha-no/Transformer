from dataclasses import dataclass


@dataclass
class Config:
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    NUM_LAYERS : int
    FEATURES : int
    NUM_HEADS : int
    FFFEATURES : int
