from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def process(self, words_list: 'typing.List[str]') -> 'np.ndarray':
        """ Processes list of words converting them into embeddings and aggregates them into one vector """
        pass
