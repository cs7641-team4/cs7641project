from interface import implements, Interface
from .LanguageModelTranslator import LanguageModelTranslator
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

class Word2Vec(implements(LanguageModelTranslator)):
    def __init__(self, path='../data/glove.6B/glove.6B.200d.txt'):
        self.model = None
        self.path = path

        self.load_model()
        pass
    

    def featurize(self, data, preprocessor, mode='multilabel', remove_unlableled=True, text_key='abstract', primary_label_key='label1', secondary_label_key='label2', neg_sample_class=11):
        features = []
        labels = []
        for row in tqdm(data):

            if remove_unlableled == True and row[primary_label_key] == '':
                continue

            abstract = row[text_key]
            abstract_clean = preprocessor.clean(abstract)
            features.append(self.convert(abstract_clean, method='average'))

            if mode.lower() == 'multilabel':
                labels.append([int(row[primary_label_key]), None if row[secondary_label_key] == '' else int(row[secondary_label_key])])
            elif mode.lower() == 'multiclass':
                labels.append([int(row[primary_label_key])])
            elif mode.lower() == 'binary':
                labels.append([0 if int(row[primary_label_key]) != neg_sample_class else 1])
            else:
                print("ERROR: unreconized mode %s. Choose one of multilabel, multiclass, or binary (0=in conference, 1=not in conference)" % mode)
            

        features = np.array(features)
        labels = np.array(labels)
        return features, labels

    def convert(self, words_list, method='average'):
        vec = []
        for word in words_list:
            try:
                vec.append(self.model[word.lower()])
            except:
                continue
        
        if method == 'average':
            vec = np.array(vec).mean(axis=0)
            
        else:
            raise NotImplementedError
            
        return vec
    
    def load_model(self):
        self.model = {}
        f = open(self.path, encoding='utf8')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32') 
            self.model[word] = coefs
        f.close()
        
        