from document import Document
from models import MockModel

import os
import re
import spacy
from tqdm import tqdm
import numpy as np
from collections import defaultdict


class Embeder:
    def __init__(self, corpus_dir):
        self._data = self._load_data(corpus_dir)

    def _load_data(self, corpus_dir):
        data = {}
        nlp = spacy.load('xx')
        for doc_name in tqdm(os.listdir(corpus_dir), desc='Loading corpus...'):
            doc_path = os.path.join(corpus_dir, doc_name)
            doc = Document.from_file(doc_path, nlp)
            # print(doc._doc)
            # print(doc._annotations)
            data[doc_name] = doc
        return data

    def get_embeddings(self, context, neighborhood, model):
        assert context in ['one-mention', 'document', 'corpus']
        embeddings = defaultdict(list)
        for _, doc in tqdm(self._data.items(), desc='Computing embeddings...'):
            neighborhood_dict = doc.get_neighbors(neighborhood)  # return words in specified neighbourhood
            for (person, category), doc_neighbors in neighborhood_dict.items():
                one_mention_embeds = np.vstack([model.process(one_mention_neighbors) for one_mention_neighbors in doc_neighbors])
                if context == 'one-mention':
                    embeddings[(person, category)].append(one_mention_embeds)
                else:
                    doc_embeds = np.mean(one_mention_embeds, axis=0)
                    embeddings[(person, category)].append(doc_embeds)  # Becomes doc embeds
        if context == 'corpus':
            for (person, category), doc_embeds in embeddings.items():
                embeddings[(person, category)] = np.mean(np.vstack(doc_embeds), axis=0)
        return embeddings

    def save_to_tensorboard(self, embeddings):
        pass


if __name__ == '__main__':
    corpus_dir = '../categorization/learningData/korpusONET'
    x = Embeder(corpus_dir)
    model = MockModel()
    print(len(x.get_embeddings('corpus', 'sentence', model)))