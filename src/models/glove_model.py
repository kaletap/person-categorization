from .base_model import BaseModel

import numpy as np
from typing import List
from gensim.models import KeyedVectors
import unicodedata


MODEL_DIR = "../../glove/glove_100_3_polish.txt"
GLOVE_SIZE = 100


class GloveModel(BaseModel):
    def __init__(self):
        print("Loading GloVe vectors...")
        self.word2vec = KeyedVectors.load_word2vec_format(MODEL_DIR)
        print("Loaded GloVe vectors.")

    @staticmethod
    def remove_polish_chars(word):
        """
        Copyright: Tymoteusz Makowski
        """
        unicodedata.normalize('NFKD', word).encode("ascii", "ignore").decode("ascii")

    def process(self, sentences: List[List[str]]) -> List[np.ndarray]:

        def get_embedding(word):
            try:
                return self.word2vec.get_vector(word)
            except KeyError:
                try:
                    return self.word2vec.get_vector(self.remove_polish_chars(word))
                except KeyError:
                    return np.zeros(100, dtype=np.float32)

        return [np.mean(np.array([get_embedding(word) for word in sentence]), axis=0) for sentence in sentences]


if __name__ == "__main__":
    glove = GloveModel()
    glove.process(["mam na imie tomek".split(), "lubie pizze".split()])


