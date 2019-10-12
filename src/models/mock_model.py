from .base_model import BaseModel

import numpy as np


class MockModel(BaseModel):
    def process(self, sentence):
        return np.random.random(size=(len(sentence), 8))
