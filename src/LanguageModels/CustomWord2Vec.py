from interface import implements, Interface
from .LanguageModelTranslator import LanguageModelTranslator
from .Word2Vec import Word2Vec
import numpy as np
import pandas as pd
import pickle
from tqdm.notebook import tqdm


class CustomWord2Vec(Word2Vec):
    def __init__(self, path='../data/customw2v.p'):
        self.model = None
        self.path = path

        self.load_model()
        pass
    
    def load_model(self):
        self.model = {}
        self.model = pickle.load(open(self.path, 'rb'))
        
        