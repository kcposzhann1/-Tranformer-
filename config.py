# config.py
import torch
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "data/train.xlsx"
    TEST_PATH = "data/test.xlsx"
    VOCAB_PATH = "vocab.txt"
    MODEL_SAVE_PATH = "model_saved.pth"
    VOCAB_SIZE = 20000
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 32
    EMBED_SIZE = 512
    FFN_HIDDEN_SIZE = 1024
    NUM_HEADS = 8
    NUM_LAYERS = 4
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    RANDOM_SEED = 42

    def __init__(self):
        self._validate()

    def _validate(self):
        if self.VOCAB_SIZE <= 0:
            raise ValueError("VOCAB_SIZE must be positive.")
        if self.MAX_SEQ_LEN <= 0:
            raise ValueError("MAX_SEQ_LEN must be positive.")
        if self.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive.")
        if self.EMBED_SIZE <= 0:
            raise ValueError("EMBED_SIZE must be positive.")
        if self.FFN_HIDDEN_SIZE <= 0:
            raise ValueError("FFN_HIDDEN_SIZE must be positive.")
        if self.NUM_HEADS <= 0:
            raise ValueError("NUM_HEADS must be positive.")
        if self.NUM_LAYERS <= 0:
            raise ValueError("NUM_LAYERS must be positive.")
        if not (0 <= self.DROPOUT <= 1):
            raise ValueError("DROPOUT must be between 0 and 1.")
        if self.LEARNING_RATE <= 0:
            raise ValueError("LEARNING_RATE must be positive.")
        if self.EPOCHS <= 0:
            raise ValueError("EPOCHS must be positive.")


